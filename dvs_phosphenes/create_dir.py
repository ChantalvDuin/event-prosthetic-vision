import os
import shutil

nr_ses = 11
nr_obj = 50
nr_fr = 300

for ses in range(1, nr_ses+1) :
    print("Start session :" + str(ses))
    for obj in range(1,nr_obj+1):
            for fr in range(0,nr_fr-1):
                    input_folder = '/home/chadui/data/dvs_phosphenes/core50_350x350_img/s%d/o%d/' % (ses, obj)
                    in_filename = "core_%0.2d_%0.2d_%0.3d.jpeg" % (ses, obj, fr)
                    in_img = input_folder + in_filename


                    if (ses == 3) | (ses == 7) | (ses == 10):
                        out_fil = '/home/chadui/data/dvs_phosphenes/core50_repr_test/s%d/o%d/' % (ses, obj) + in_filename

                    else:
                        out_fil = '/home/chadui/data/dvs_phosphenes/core50_repr_train/s%d/o%d/' % (ses, obj) + in_filename

