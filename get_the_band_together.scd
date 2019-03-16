//// Magical Music Machine :: Mapping Motion to Music :: MMM
//// HWSW Lab 1 : Winter 2019
//// SC Code by Andrew Luck
//// Updated: 3/13/2019

// Dependencies: Quarks.new("https://github.com/supercollider-quarks/wslib");

s.boot; // boot audio server

// Drum Kit*************************************
// *********************************************

// Bring in full folder of samples
(
  // Run everything inside this paren to initialize

// Load buffers********************************

~drum_kitSF = Array.new;
~folder = PathName.new("C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/drum_kit");

~folder.entries.do({
	arg path;
	~drum_kitSF = ~drum_kitSF.add(Buffer.read(s, path.fullPath));
});



// Define synths*********************************

~airhorn = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/fx/airhorn.wav");
// ~airhorn.play;

~goat = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/fx/goat.wav");
// ~goat.play;

~bow = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/fx/bow.wav");
// ~bow.play;

~crowd = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/fx/crowd.wav");
//~crowd.play;

~hornstab = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/fx/hornstab.wav");
// ~hornstab.play;

~dog = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/fx/dog.wav");
// ~dog.play;

SynthDef.new(\hornstab, {
	arg amp = 1, out = 0, buf, da = 4, rate = 1, pan = 0, atk = 0.001, rel = 0.7, midinote = 36, midiratio = 1;
	var sig, env, loudness;
	loudness = 3.0;
	rate = (midinote - 60).midiratio;
	sig = PlayBuf.ar(2, buf, rate, doneAction:da);
	env = EnvGen.kr(Env.new([0, 1, 0], [atk, rel*3], [1, -1]), doneAction: 2);
	sig = Pan2.ar(sig, pan, amp);
	sig = sig * env * loudness;
	Out.ar(out, sig);
}).add;

// Synth.new(\hornstab, [\midinote, ~guitarNotes.choose, \buf, ~hornstab, \amp, 0.1]);


SynthDef.new(\playdrums, {
	arg amp = 0.7, out = 0, buf, da = 2, rate = 1, midinote = 1;
	var sig;
	midinote.postln;
	buf = midinote - 36;
	midinote.postln;
	sig = PlayBuf.ar(1, buf, rate, doneAction:da);
	sig = sig * amp;
	sig = Splay.ar(sig) * 0.5;
	Out.ar(out, sig);
}).add;

SynthDef.new(\playdrums2, {
	arg amp = 0.7, out = 0, buf, da = 2, rate = 1, midinote = 1;
	var sig;
	midinote.postln;
	midinote.postln;
	sig = PlayBuf.ar(1, buf, rate, doneAction:da);
	sig = sig * amp;
	sig = Splay.ar(sig) * 0.5;
	Out.ar(out, sig);
}).add;


// Synth.new(\playdrums2, [\buf, ~drum_kitSF.choose]);

~piano = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/piano.wav");
// Synth.new(\playpiano, [\midinote, ~guitarNotes.choose, \buf, ~piano, \amp, 1.0]);

SynthDef.new(\playpiano, {
	arg amp = 0.7, out = 0, buf, da = 4, rate = 1, pan = 0, atk = 0.005, rel = 0.7, midinote = 36, midiratio = 1;
	var sig, env, loudness;
	loudness = 2.0;
	rate = (midinote - 60).midiratio;
	sig = PlayBuf.ar(2, buf, rate, doneAction:da);
	env = EnvGen.kr(Env.new([0, 1, 0], [atk, rel*3], [1, -1]), doneAction: 2);
	sig = Pan2.ar(sig, pan, amp);
	sig = sig * env * loudness;
	Out.ar(out, sig).distort;
}).add;


// Guitar*************************************
// *******************************************

~guitarSF = Buffer.read(s, "C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/guitar_midC_60.wav");
// ~guitarSF.play;

SynthDef.new(\playguitar, {
	arg amp = 0.7, out = 0, buf, da = 4, rate = 1, pan = 0, atk = 0.005, rel = 0.7, midinote = 36, midiratio = 1;
	var sig, env, loudness;
	loudness = 2.0;
	rate = (midinote - 60).midiratio;
	sig = PlayBuf.ar(2, buf, rate, doneAction:da);
	env = EnvGen.kr(Env.new([0, 1, 0], [atk, rel*3], [1, -1]), doneAction: 2);
	sig = Pan2.ar(sig, pan, amp);
	sig = sig * env * loudness;
	Out.ar(out, sig).distort;
}).add;


// Synth.new(\playguitar, [\midinote, ~guitarNotes.choose, \buf, ~guitarSF, \amp, 0.1]);



// Bass*************************************
// *****************************************

~bassguitar = Buffer.read(s,"C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/bassguitar2_midC_60.wav");
// ~bassguitar.play;


SynthDef.new(\playbassguitar, {
	arg amp = 0.9, out = 0, buf, da = 2, rate = 1, pan = 0, atk = 0.005, rel = 0.9, midinote = 1, loudness = 10;
	var sig, env;
	rate = (midinote - 36).midiratio;
	sig = PlayBuf.ar(2, buf, rate, doneAction:da);
	env = EnvGen.kr(Env.new([0, 1, 0], [atk, rel], [1, -1]), doneAction: 2);
	sig = Pan2.ar(sig, pan, amp);
	sig = sig * env * loudness;
	Out.ar(out, sig);
}).add;


// Patterning*******************************************************************************
// *****************************************************************************************


// Bring in Magenta MIDIs

// import 16 bar song parts
~midiFileNumber = (500).rand;
"****************************".post;
"Song # is:".postln;
~midiFileNumber.postln;
// ~midiFileNumber = (~midiFileNumber - (~midiFileNumber%2));
// ~midiFileNumber2 = ~midiFileNumber + 1;
m = SimpleMIDIFile.read("C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/midi/"++~midiFileNumber++".mid");
// ~m2 = SimpleMIDIFile.read("C:/Users/joshu/OneDrive/Documents/GIX/MMM_314/midi/"++~midiFileNumber2++".mid");

// // import bridge parts
// ~m3 = SimpleMIDIFile.read("C:/Users/cl/Documents/MMM/midi/~midiFileNumber/"1"++".mid");
// ~m4 = SimpleMIDIFile.read("C:/Users/cl/Documents/MMM/midi/~midiFileNumber/"2"++".mid");
// ~m5 = SimpleMIDIFile.read("C:/Users/cl/Documents/MMM/midi/~midiFileNumber/"3"++".mid");
// ~m6 = SimpleMIDIFile.read("C:/Users/cl/Documents/MMM/midi/~midiFileNumber/"4"++".mid");


// 16 bar song parts a and b as arrays
a = m.generatePatternSeqs;
// b = ~m2.generatePatternSeqs;

// // 4 bar lerps as arrays
// c = ~m3.generatePatternSeqs;
// d = ~m4.generatePatternSeqs;
// e = ~m5.generatePatternSeqs;
// f = ~m6.generatePatternSeqs;


~currentSongPart = a;

// create new array with notes from guitar sequence
~guitarNotes = Array.new(maxSize: ~currentSongPart[1].size);
~currentSongPart[1].do({arg item, i;
	if(	~currentSongPart[1][i][0] != 'rest',
		{
			~guitarNotes.add(~currentSongPart[1][i][0]);
			[i].postln;
	})
});


~currentSongPart[1].do({arg item, i;
			~guitarNotes.add(~currentSongPart[1][i][0]);
	});

~guitarNotes.size;
~guitarNotes.maxSize;
~guitarNotes;



// create new array with notes from guitar sequence
~bassGuitarNotes = Array.new(maxSize: ~currentSongPart[1].size);

~currentSongPart[2].do({arg item, i;
	if(	~currentSongPart[2][i][0] != 'rest',
		{
			~bassGuitarNotes.add(~currentSongPart[2][i][0]);
			// [i].postln;
	})
});



~currentSongPart[1].postln;  // guitar
~currentSongPart[2].postln;  // bass
~currentSongPart[3].postln;  // drums

// // turn guitar notes into power chords
// ~currentSongPart[1].do({arg item, i;
// 	~currentSongPart[1][i][0] = [~currentSongPart[1][i][0] -5, ~currentSongPart[1][i][0]-12, ~currentSongPart[1][i][0]];
// });

// Time Clock ***************************************************************************
// **************************************************************************************


~tempo = (rand(30) + 90)/60;
~tempo;
// ~tempo = 0.90;
// TempoClock.default.tempo = ~tempo;
t= TempoClock.new(~tempo);

~postInfo = {
	("beat : "+(((t.beats)%t.beatsPerBar)+1)).postln;
	("bar : "+(t.bar)).postln;
	"".postln;
	1;
};

t.schedAbs(t.nextBar, {
	~postInfo.value;
});

)

// RIFFS ********************************************************************************
// **************************************************************************************

// Long guitar note / big strum

Synth.new(\playguitar, [\midinote, ~guitarNotes[0..7].choose, \buf, ~guitarSF, \amp, 0.1]);


// Long bass not / big strum

Synth.new(\playbassguitar, [\midinote, ~bassGuitarNotes[0..3].choose, \buf, ~bassguitar, \amp, 0.7]);


// Slow guitar picking
(
Pbind(
		\instrument, \playguitar,
		\buf, ~guitarSF,
	\dur, Pseq([1/4], inf),
	\rel, Pxrand([0.2:0.5], inf),
	\midinote, Pxrand(~guitarNotes , inf)).play;
)


// Medium guitar picking
(
Pbind(
		\instrument, \playguitar,
		\buf, ~guitarSF,
	\dur, Pseq([1/8], inf),
	\rel, Pxrand([0.2:0.5], inf),
		\midinote, Pxrand(~guitarNotes +12 , inf)).play;
)

// Fast guitar shredding
(
~shred = Pdef(\shred, Pbind(
		\instrument, \playguitar,
		\buf, ~guitarSF,
	\dur, Pseq([1/16], inf),
	\rel, Pxrand([0.1:0.2], inf),
	\atk, Pxrand([0.005:0.05], inf),
	\midinote, Pxrand(~guitarNotes + 12 , inf))).play;
)

// ~shred.stop;

// Synth.new(\playguitar, [\midinote, ~guitarNotes.choose, \buf, ~guitarSF, \amp, 0.1]);


// Google Magenta AI Trio : guitar / bass / drum sequence together


(
g = Pdef(\guitarz, Pbind(
			\instrument, \playguitar,
			\buf, ~guitarSF,
			\strum, 0.025,
			[\midinote, \dur],
			Pseq(~currentSongPart[1], 1))).play;
b = Pdef(\bassz, Pbind(
			\instrument, \playbassguitar,
			\buf, ~bassguitar,
			[\midinote, \dur],
			Pseq(~currentSongPart[2], 1))).play;
d =	Pdef(\drumz, Pbind(
	\instrument, \playdrums,
			\buf, ~drum_kitSF,
			[\midinote, \dur],
			Pseq(~currentSongPart[3], 1))).play;
);





//OSCDefinitions ************************************************************************
// **************************************************************************************

(
OSCdef.new(
	\bang,
	{
		arg msg, time, addr, port;
		msg.postln;
		msg[1].postln;
		if(msg[1] == 'bang', {
			~init;
			~crowd.play;
			g = Pdef(\guitarz, Pbind(
				\instrument, \playguitar,
				\buf, ~guitarSF,
				\strum, 0.025,
				\amp, 0.1,
				[\midinote, \dur],
				Pseq(~currentSongPart[1], 1))).play;
			b = Pdef(\bassz, Pbind(
				\instrument, \playbassguitar,
				\buf, ~bassguitar,
				[\midinote, \dur],
				Pseq(~currentSongPart[2], 1))).play;
			d =	Pdef(\drumz, Pbind(
				\instrument, \playdrums,
				\buf, ~drum_kitSF,
				[\midinote, \dur],
			Pseq(~currentSongPart[3], 1))).play;
			});
		if(msg[1] == 'bigstrumguitar', {
			Synth.new(\playguitar, [\midinote, ~guitarNotes.choose, \buf, ~guitarSF, \amp, 0.1]);
		});
		if(msg[1] == 'bigstrumbass', {
			Synth.new(\playbassguitar, [\midinote, ~bassGuitarNotes[0..3].choose, \buf, ~bassguitar, \amp, 0.7]);
		});
		if(msg[1] == 'highkick', {
			Synth.new(\hornstab, [\midinote, ~guitarNotes.choose, \buf, ~hornstab, \amp, 1.5]);
		});
		if(msg[1] == 'crowd', {
			~crowd.play;
		});
		if(msg[1] == 'bow', {
			~bow.play;
		});
		if(msg[1] == 'strum', {
			Synth.new(\playguitar, [\midinote, ~guitarNotes[0..7].choose, \buf, ~guitarSF, \amp, 0.1]);
		});
		if(msg[1] == 'drums1', {
			~drum_kitSF[11].play;  // tom
		});
		if(msg[1] == 'drums2', {
			~drum_kitSF[12].play;  // tom
		});
		if(msg[1] == 'drums3', {
			~drum_kitSF[10].play;  // open hat
		});
		if(msg[1] == 'drums4', {
			~drum_kitSF[14].play;  // crash
		});
		if(msg[1] == 'drums5', {
			~drum_kitSF[5].play;  // open hat
		});
		if(msg[1] == 'drums6', {
			~drum_kitSF[9].play;  // open hat
		});
		if(msg[1] == 'cowbell', {
			~drum_kitSF[3].play;  // cowbell
		});
		if(msg[1] == 'dab', {
			~airhorn.play;  // cowbell
		});
		if(msg[1] == 'goats', {
			~goat.play;  // goat
		});
		if(msg[1] == 'piano', {
			Synth.new(\playpiano, [\midinote, ~guitarNotes.choose, \buf, ~piano, \amp, 1.0]);
		});
		if(msg[1] == 'dog', {
			~dog.play;  // goat
		});
			},
	'/bang'
);
)



OSCFunc.trace(true); // Turn posting on
OSCFunc.trace(false); // Turn posting off
