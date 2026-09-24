"""
experiment_stimulus.py
-----------------------
PsychoPy Coder script that runs the Graz-style Motor-Imagery (MI) trial
protocol described in the tutorial slide "The Graz Motor-Imagery trial
protocol", and pushes an LSL marker at every event so the recording can be
aligned with the EEG stream during analysis (see slide "Synchronizing
devices with Lab Streaming Layer").

Trial timeline (one trial ~ 8 s):
    0-2 s   Fixation   - a white cross, subject relaxes
    2 s     Cue        - a left/right arrow + a short beep
    2-6 s   Imagery    - arrow stays on screen, subject imagines the move
    6-8 s   Rest / ITI - blank screen before the next trial

Run alongside:
    - the amplifier's own LSL EEG outlet (or `send_eeg_data.py` to simulate
      one), and
    - LabRecorder, to save both streams into a single synchronized .xdf file.

Usage:
    python experiment_stimulus.py --n-trials 40 --block-name run1
"""
import argparse
import random

from psychopy import visual, core, event, sound
from pylsl import StreamInfo, StreamOutlet


def build_condition_list(n_trials):
    """fullRandom-style condition list: balanced left/right, shuffled."""
    assert n_trials % 2 == 0, "n_trials must be even for a balanced design"
    conditions = ["left"] * (n_trials // 2) + ["right"] * (n_trials // 2)
    random.shuffle(conditions)
    return conditions


def make_marker_outlet(source_id="mi-tutorial-markers"):
    info = StreamInfo(
        name="MI_Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,          # irregular rate: markers are event-driven
        channel_format="string",
        source_id=source_id,
    )
    return StreamOutlet(info)


def run_block(win, outlet, conditions, timings, beep):
    fixation = visual.TextStim(win, text="+", height=0.15, color="white")
    left_arrow = visual.TextStim(win, text="←", height=0.25, color="white")
    right_arrow = visual.TextStim(win, text="→", height=0.25, color="white")
    clock = core.Clock()

    for i, cond in enumerate(conditions):
        arrow = left_arrow if cond == "left" else right_arrow

        # --- Fixation -------------------------------------------------
        outlet.push_sample([f"trial_{i:03d}_fixation_start"])
        fixation.draw()
        win.flip()
        core.wait(timings["fixation"])

        # --- Cue (beep + arrow appears) --------------------------------
        beep.play()
        outlet.push_sample([f"trial_{i:03d}_cue_{cond}"])
        arrow.draw()
        win.flip()
        core.wait(timings["cue"])

        # --- Motor imagery window --------------------------------------
        outlet.push_sample([f"trial_{i:03d}_imagery_start"])
        arrow.draw()
        win.flip()
        core.wait(timings["imagery"])
        outlet.push_sample([f"trial_{i:03d}_imagery_end"])

        # --- Rest / inter-trial interval --------------------------------
        win.flip()  # blank screen
        core.wait(timings["rest"])

        # Allow the experimenter to abort with Escape between trials
        if "escape" in event.getKeys():
            outlet.push_sample(["experiment_aborted"])
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40,
                         help="Total trials in this run (balanced left/right).")
    parser.add_argument("--block-name", type=str, default="run1",
                         help="Label pushed as the block-start marker.")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    timings = {"fixation": 2.0, "cue": 2.0, "imagery": 4.0, "rest": 2.0}

    win = visual.Window(fullscr=args.fullscreen, color="black", units="height")
    beep = sound.Sound("A", secs=0.15)
    outlet = make_marker_outlet()

    conditions = build_condition_list(args.n_trials)

    outlet.push_sample([f"block_start_{args.block_name}"])
    instructions = visual.TextStim(
        win,
        text=(f"Block: {args.block_name}\n\n"
              f"{args.n_trials} trials — imagine squeezing the cued hand\n"
              f"until the arrow disappears.\n\nPress any key to start."),
        height=0.06, color="white", wrapWidth=1.4,
    )
    instructions.draw()
    win.flip()
    event.waitKeys()

    run_block(win, outlet, conditions, timings, beep)

    outlet.push_sample([f"block_end_{args.block_name}"])
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
