k0 = [9894780.01] # for 635nm light = 2*pi/lamb 
n_para = 1.7
n_perp = 1.5

processor1 = LCProcessor.LCProcessor(n_para, n_perp, ks=k0)
processor1.parse_dataframe_from_txt(director_data.txt) # parsing data from text

processor1.generate_jones()
processor1.generate_flattened_jones()

processor1.plot_slice_of_director_field()
processor1.plot_intensity_image(rotated_cross_pol_angle=45*math.pi/180)
processor1.plot_intensity_video()

k0 = [9894780.01] # for 635nm light = 2*pi/lamb 
n_para = 1.7
n_perp = 1.5

processor2 = LCProcessor.LCProcessor(n_para, n_perp, ks=k0)
processor2.create_defect_dataframe(20, 1.5e-8) # generating data from defect

processor2.generate_jones()
processor2.generate_flattened_jones()

processor2.plot_slice_of_director_field()
processor2.plot_intensity_image()
processor2.plot_intensity_video()