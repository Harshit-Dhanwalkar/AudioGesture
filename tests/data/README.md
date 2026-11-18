Recorded hand traces for the regression tests.

Record a session: `./build/audiogesture audio/file_example_WAV_2MG.wav --record-trace tests/data/my-moment.trace`
Replay it:        `/build/audiogesture audio/file_example_WAV_2MG.wav --replay tests/data/my-moment.trace`

`test_trace` replays every `*.trace` file in this directory through the gesture mapper and
checks the invariants that must hold for any input (finite parameters, stretch within its
limits, gains within -1..1, filters in range, nothing stuck frozen once the hands are gone).
Keep traces short (a few hundred frames) and free of anything you do not want published:
they contain only landmark coordinates, never camera images.
