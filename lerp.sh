mkdir $3
music_vae_generate \
--config=hierdec-trio_16bar \
--checkpoint_file=/Users/andrewluck/Documents/hierdec-trio_16bar.tar \
--mode=interpolate \
--num_outputs=4 \
--input_midi_1=$1 \
--input_midi_2=$2 \
--output_dir=$3
sudo mv $1 $3