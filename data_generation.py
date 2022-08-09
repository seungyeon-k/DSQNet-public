import argparse
from functions.data_generator import generate_data, save_data

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=int, default=0)
	args = parser.parse_args()

	object_names = ['box', 'cylinder', 'ellipsoid']
	dir_name = 'datasets/example_dataset'
	num_partial_pnts = 300
	save_full_pc = True
	num_objects = 10
	append = True
	num_cams = 16

	for trial in range(1000):
		success = generate_data(object_names, 
								num_objects, 
								num_cams, 
								num_pnts=num_partial_pnts, 
								append=append, 
								save_full_pc=save_full_pc, 
								save_membership=True, 
								plot_membership=False, 
								visualize_pc=False, 
								visualize_pc_with_mesh=False, 
								render_mesh=False, 
								dir_name=dir_name)
		if success:
			object_names.pop(0)
		if not object_names:
			break

	# split
	save_data(
			default_path = dir_name,
			shuffle = False, 
			train_test_ratio = 0.9,
			train_val_ratio = 0.9
	)

