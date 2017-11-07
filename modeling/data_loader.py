import numpy as np
##
##	Split the given song into num_seg segments, 
##  each of length win_len and overlap of 
##	(win_len - win_hop)
##
def split_song(audData, win_len=300, win_hop=100, num_seg=20, num_freq_comp=120):
	split_segments = np.zeros((num_seg, num_freq_comp, win_len))
	for seg in range(num_seg):
		split_segments[seg] = audData[:,int(win_hop*seg):int(win_hop*seg+win_len)]
	return split_segments


def songs2batch(data, label, num_seg=20, batch_size=40, num_genre=8, inp_dim=(120, 300)):
	num_parts =  num_seg*num_genre // batch_size		## divide a song with num_seg segments into num_parts parts
	batches = np.zeros((num_parts, batch_size, *inp_dim))
	data_split = np.hsplit(data, num_parts)
	for part in range(num_parts):
		batches[part] = data_split[part].reshape(batches.shape[1:])
	labels = np.eye(num_genre)[np.repeat(label, int(num_seg//num_parts))]
	return batches, labels

def get_batches(paths, labels, is_train, num_classes=8):
	all_songs = []
	data_batch = []
	batch_size = 40
	num_seg = 20

	## Test Case
	## Each song is a single batch
	if is_train is not True:
		for i,song_path in enumerate(paths):
			audData = np.load(song_path)
			split_seg = split_song(audData, num_seg=num_seg)
			label_song = np.eye(num_classes)[np.repeat(labels[i], num_seg)]
			data_batch.append({'X':split_seg, 'y':label_song})
		return data_batch

	for cl in range(num_classes):
		audData = np.load(paths[cl])
		all_songs.append(split_song(audData, num_seg=num_seg))

	batch_data, batch_label = songs2batch(np.array(all_songs), labels, num_seg=num_seg, 
			batch_size=batch_size, num_genre=num_classes)


	for i in range(len(batch_data)):
		data_batch.append({'X': batch_data[i],'y': batch_label})

	return data_batch
