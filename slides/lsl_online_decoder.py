"""
lsl_online_decoder.py
-----------------------
Standalone real-time decoding loop for the Motor-Imagery BCI built in the
tutorial notebook (slides "From offline model to real-time predictions" and
the notebook's Section 6b "continuous, over-time replay" — this script runs
the exact same steps against a *live* LSL stream instead of a replayed file):

    1. Pull a chunk    - read the newest samples from the LSL EEG inlet
    2. Preprocess (a)  - causal 1 Hz high-pass, persistent filter state
                          (same broadband signal the notebook fits ICA on)
    3. Slide the window - keep a fixed-length causal buffer (e.g. 2-4 s)
    4. Clean            - apply the PRETRAINED ICA saved by the notebook
                          (`ica_weights-ica.fif`) — never refit online, just
                          re-apply the fixed spatial unmixing learned offline
    5. Preprocess (b)  - causal 7-30 Hz band-pass (mu + beta) on the cleaned
                          window
    6. Predict          - pipeline.predict() with the model trained in the
                          notebook (csp_lda_pipeline.joblib)
    7. Act              - push the predicted class onward as a new LSL
                          stream, so a feedback / stimulus program can
                          consume it

Run this alongside an EEG LSL outlet (a real amplifier, or `send_eeg_data.py`
to simulate one) and, optionally, `experiment_stimulus.py` for markers.

Usage:
    python lsl_online_decoder.py \
        --model csp_lda_pipeline.joblib --ica ica_weights-ica.fif \
        --window-sec 2.0 --step-sec 0.5

Note: the incoming LSL stream's channels are assumed to be the same
channels, in the same order, that the model and ICA were trained on (this is
what `ica.ch_names` records). A real amplifier stream should be configured
to match, or the script should be extended to reorder/pick channels by name.
"""
import argparse
import time
from collections import deque

import numpy as np
import joblib
import mne
from scipy.signal import butter, lfilter, lfilter_zi
from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop

mne.set_log_level("ERROR")


def make_causal_filter(freq, sfreq, order=4, n_channels=32, btype="highpass"):
    """A causal (online-safe), stateful IIR filter: consecutive chunks are
    filtered continuously (the internal state carries over), instead of each
    chunk starting from zero — which would create an edge artifact at every
    chunk boundary. Used for both the 1 Hz high-pass (step 2) and would work
    the same way for a persistent band-pass; the narrow 7-30 Hz band-pass
    here is instead applied fresh per *window* (step 5), since ICA (step 4)
    is only run once a full window is available — see `predict_on_window`.
    """
    b, a = butter(order, freq, btype=btype, fs=sfreq)
    zi_single = lfilter_zi(b, a)
    zi = np.tile(zi_single, (n_channels, 1)).T  # shape (order, n_channels)
    state = {"b": b, "a": a, "zi": zi}

    def apply(chunk):
        # chunk: (n_samples, n_channels)
        filtered, state["zi"] = lfilter(state["b"], state["a"], chunk, axis=0,
                                         zi=state["zi"])
        return filtered

    return apply


def causal_bandpass_window(window, low, high, sfreq, order=4):
    """Causal 7-30 Hz band-pass applied to a single (n_channels, n_times)
    window. Initializes the filter's state from the window's own first
    sample (the standard `lfilter_zi * x[:, :1]` trick) to approximate a
    steady state and reduce the start-of-window transient, since — unlike
    the high-pass in `make_causal_filter` — this filter does not carry state
    across successive (overlapping) windows."""
    b, a = butter(order, [low, high], btype="bandpass", fs=sfreq)
    zi = lfilter_zi(b, a)
    zi = np.outer(window[:, 0], zi)  # (n_channels, order), per-channel steady state
    filtered, _ = lfilter(b, a, window, axis=1, zi=zi)
    return filtered


def apply_ica_to_window(ica, window, ch_names, sfreq):
    """Apply the PRETRAINED ICA (fit offline, in the notebook) to one
    (n_channels, n_times) window. ICA is a fixed spatial transform with no
    temporal memory, so applying it window-by-window (rather than
    continuously) gives identical results to applying it to the whole
    stream — no edge effects to worry about here, unlike the IIR filters."""
    info = mne.create_info(ch_names, sfreq, "eeg")
    raw_window = mne.io.RawArray(window, info, verbose=False)
    ica.apply(raw_window, verbose=False)
    return raw_window.get_data()


def connect_inlet(stream_type="EEG", timeout=10.0):
    print(f"Resolving LSL stream of type '{stream_type}'...")
    streams = resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        raise RuntimeError(
            f"No LSL stream of type '{stream_type}' found within {timeout}s. "
            "Is the amplifier / send_eeg_data.py running?"
        )
    inlet = StreamInlet(streams[0], max_buflen=60)
    info = inlet.info()
    print(f"Connected to '{info.name()}'  ({info.channel_count()} ch, "
          f"{info.nominal_srate()} Hz)")
    return inlet, info


def make_prediction_outlet(source_id="mi-tutorial-online-decoder"):
    info = StreamInfo(
        name="MI_Prediction",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id=source_id,
    )
    return StreamOutlet(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="csp_lda_pipeline.joblib",
                         help="Path to the pipeline saved from the notebook.")
    parser.add_argument("--ica", default="ica_weights-ica.fif",
                         help="Path to the pretrained ICA saved from the "
                              "notebook. Pass --no-ica to skip cleaning.")
    parser.add_argument("--no-ica", action="store_true",
                         help="Skip ICA cleaning (predict on filtered-only data).")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--highpass-hz", type=float, default=1.0,
                         help="Broadband high-pass applied before ICA (step 2).")
    parser.add_argument("--low-hz", type=float, default=7.0)
    parser.add_argument("--high-hz", type=float, default=30.0)
    parser.add_argument("--stream-type", default="EEG")
    args = parser.parse_args()

    model = joblib.load(args.model)
    print(f"Loaded pipeline: {model}")

    ica = None
    if not args.no_ica:
        ica = mne.preprocessing.read_ica(args.ica, verbose=False)
        print(f"Loaded pretrained ICA ({len(ica.ch_names)} channels, "
              f"{len(ica.exclude)} component(s) excluded) — will apply to "
              "every window, without refitting.")

    inlet, info = connect_inlet(stream_type=args.stream_type)
    sfreq = info.nominal_srate()
    n_channels = info.channel_count()
    ch_names = ica.ch_names if ica is not None else [f"ch{i}" for i in range(n_channels)]

    expected_ch = getattr(getattr(model, "named_steps", {}).get("csp", None), "n_channels_", None)
    if expected_ch is not None and expected_ch != n_channels:
        print(f"WARNING: model was trained on {expected_ch} channels, "
              f"this stream has {n_channels}. Predictions will be unreliable "
              "until the channel counts match (e.g. don't use the "
              "send_eeg_data.py demo stream with a model trained on real EEG).")

    # Step 2: causal broadband high-pass, persistent state across chunks
    highpass = make_causal_filter(args.highpass_hz, sfreq, n_channels=n_channels,
                                   btype="highpass")
    pred_outlet = make_prediction_outlet()

    window_len = int(args.window_sec * sfreq)
    buffer = deque(maxlen=window_len)
    label_names = {0: "left", 1: "right"}

    print(f"Streaming: {args.window_sec}s window, predicting every "
          f"{args.step_sec}s. Ctrl+C to stop.")

    next_predict_time = time.time()
    try:
        while True:
            # 1. Pull whatever chunk is available right now
            chunk, timestamps = inlet.pull_chunk(timeout=0.2)
            if chunk:
                chunk = np.asarray(chunk)              # (n_samples, n_channels)
                broadband = highpass(chunk)             # 2. causal high-pass
                buffer.extend(broadband.tolist())       # 3. slide the window

            # 4-6. Once the buffer is full and the step interval elapsed
            now = time.time()
            if len(buffer) == window_len and now >= next_predict_time:
                window = np.array(buffer).T             # (n_channels, n_times)

                if ica is not None:
                    window = apply_ica_to_window(ica, window, ch_names, sfreq)  # 4.

                window = causal_bandpass_window(window, args.low_hz, args.high_hz, sfreq)  # 5.

                pred = model.predict(window[np.newaxis, ...])[0]               # 6.
                label = label_names.get(int(pred), str(pred))

                pred_outlet.push_sample([label])                               # 7.
                print(f"[{time.strftime('%H:%M:%S')}] prediction -> {label}")

                next_predict_time = now + args.step_sec
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
