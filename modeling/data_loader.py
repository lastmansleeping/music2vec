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


def create_batches(data, num_seg=20, batch_size=40, num_genre=8, inp_dim=(120, 300)):
	num_parts =  num_seg*num_genre // batch_size		## divide a song with num_seg segments into num_parts parts
	batches = np.zeros((num_parts, batch_size, *inp_dim))
	print(batches.shape)
	data_split = np.hsplit(data, num_parts)
	for part in range(num_parts):
		batches[part] = data_split[part].reshape(batches.shape[1:])

	return batches

if __name__ == '__main__':
	all_songs = []
	num_genre = 8
	path = '/home/rabbeh/Projects/DL/Data/npy/'
	np_files = ['100552.npy','100949.npy','100975.npy','113558.npy']*2
	for genre_ind in range(num_genre):
		audData = np.load(path + np_files[genre_ind])
		all_songs.append(split_song(audData))

	batches = create_batches(np.array(all_songs))

