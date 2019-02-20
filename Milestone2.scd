// OSCFunc.trace(true); // Turn posting on
// OSCFunc.trace(false); // Turn posting off

// This script recieves OSC from python via 127.0.0.1 @ port 57120 and triggers buffers
// stored on the Bela

// Dependencies: Install and compile bela-remote SynthDef classes to Extensions directory // in local user support directory.

// On Bela: Start and run the main.scd file in the Milestone2 project of the Bela IDE

// Initialize the buffers and defs on file open
(
~b0 = Buffer.read(s, "cowbell.wav");
~b0.duration;
~b1 = Buffer.read(s, "drums.wav");
~b1.duration;
~b2 = Buffer.read(s, "guitar.wav");

(
OSCdef.new(
	\bang,
	{
		arg msg, time, addr, port;
		msg.postln;
		msg[1].postln;
		if(msg[1] == 'cowbell', {
			SynthDef(\cowbell, {| out = 0, bufnum = 0 |
			Out.ar(out,
			PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction: Done.freeSelf)
			)
			}).play(s, [\out, [0, 1], \bufnum, ~b0]);
		});
		if(msg[1] == 'drums', {
			SynthDef(\guitar, {| out = 0, bufnum = 0 |
			Out.ar(out,
			PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction: Done.freeSelf)
			)
			}).play(s, [\out, [0, 1], \bufnum, ~b1]);
		});
		if(msg[1] == 'guitar', {
			SynthDef(\guitar, {| out = 0, bufnum = 0 |
			Out.ar(out,
			PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction: Done.freeSelf)
			)
			}).play(s, [\out, [0, 1], \bufnum, ~b2]);
		});
	},
	'/bang'
)
);
)