__author__ = 'hafiz'
from FeatureExtractor import *
from RF import *
from subactivities import  *
import shutil

if __name__=='__main__':
    fnam = "collection7-30"

    activity_name = get_activities()
    fobj = FeatureExtractor()
    # framelenghts=[200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
    fln = 2500
    step_sizes=[100,200,500,800,1000,1250,1500,1800]
    fp = open("Data/"+fnam+"/"+activity_name[0]+".csv", 'w')
    for stp in step_sizes:
        sys.argv = ["FeatureExtractor.py", "Data/"+fnam+"/train" ,"train"]

        # stp = int(fln/2)
        fobj.set_frameslength(fln,stp)
        fobj.main()

        sys.argv= ["python FeatureExtractor.py", "Data/"+fnam+"/eval", "test"]
        fobj.main()

        rfObj = RFClassifier()
        # rfObj.main()
        r = str(stp)
        rs = []
        rs = rfObj.main_fusion()
        for a in rs:
            r=r+","+str(a)
        l = rfObj.main()
        result=[]
        for a in l:
            r=r+","+str(a)
        result.extend(rs)
        result.extend(l)
        fp.write(r)
        fp.write('\n')

        print '\n\nRemove feature directory, csv, fusion, models....\n\n'

        dest = "Data/"+fnam+"/train/features"
        shutil.rmtree(dest, ignore_errors=True)
        dest = "Data/"+fnam+"/train/csv"
        shutil.rmtree(dest, ignore_errors=True)
        dest = "Data/"+fnam+"/train/fusion1"
        shutil.rmtree(dest, ignore_errors=True)
        dest = "Data/"+fnam+"/train/fusion2"
        shutil.rmtree(dest, ignore_errors=True)
        dest = "Data/"+fnam+"/train/fusion3"
        shutil.rmtree(dest, ignore_errors=True)

        dest = "Data/"+fnam+"/eval/features"
        shutil.rmtree(dest, ignore_errors=True)
        dest= "Data/"+fnam+"/eval/csv"
        shutil.rmtree(dest, ignore_errors=True)
        dest="Data/"+fnam+"/eval/fusion1"
        shutil.rmtree(dest, ignore_errors=True)
        dest="Data/"+fnam+"/eval/fusion2"
        shutil.rmtree(dest, ignore_errors=True)
        dest="Data/"+fnam+"/eval/fusion3"
        shutil.rmtree(dest, ignore_errors=True)

        dest= "Results/fusion/*"
        shutil.rmtree(dest, ignore_errors=True)
        dest ="Results/fusion"
        shutil.rmtree(dest, ignore_errors=True)
        dest= "Results"
        shutil.rmtree(dest, ignore_errors=True)
        os.remove('RFModel.pk')
        # break
    fp.close()
