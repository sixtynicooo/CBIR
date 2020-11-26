from packages.orb.orbdescriptor import OrbDescriptor
import glob
import cv2


import pandas as pd # type(des1) è numpy.ndarray , devo usare questa libreria per trasformare e salvare nel csv

class IndexOrb:
    def __init__(self, index_path,index_orb_kp_path,index_orb_des_path, dataset_path, progress, root,lowe_ratio,default_dataset):
        self.index_path = index_path
        self.dataset_path = dataset_path
        self.progress = progress
        self.root = root
        self.lowe_ratio=lowe_ratio
        self.index_orb_kp_path=index_orb_kp_path
        self.index_orb_des_path=index_orb_des_path

    def indexing(self):
        # inizializzazione spazio colore
        cd = OrbDescriptor()
        # apertura file output del index per la scrittura
        output_nome_dim = open(self.index_path, "w")
        output_kp=open(self.index_orb_kp_path, "w")#csv con i kp
        output_des=open(self.index_orb_des_path, "w")#csv con i des
        # calcolo del numero di immagini da processare
        total_images =len(glob.glob(self.dataset_path + "/*.jpg"))
        counter = 1
        # loop delle immagini
        i=1
        for image_path in glob.glob(self.dataset_path + "/*.jpg"):

            # estrazione del nome dell'immagine e usato come ID
            image_id = image_path[image_path.rfind("/") + 1:]
            # lettura e caricamento immagine
            image = cv2.imread(image_path,0)#in bianco e nero metto ,0
            # estrazione delle feature dalla immagine
            kp,des = cd.describe(image)
            #preparazione punti di coordinate
            for point in kp:#funziona
                p = str(point.pt[0]) + "," + str(point.pt[1]) + "," + str(point.size) + "," + str(point.angle) + "," + str(
                point.response) + "," + str(point.octave) + "," + str(point.class_id) + "\n"
                output_kp.write(p) 
            features_des = [str(f) for f in des]
            features_righe_colonne = [str(f) for f in des.shape]
            output_nome_dim.write("%s,%s\n" % (image_id,  ",".join(features_righe_colonne)))
            output_des.write("%s:\n" % ",".join(features_des))
            print(image_id)
            i+=1            
            # calcolo della percentuale di progresso e aggiornamento barra
            self.progress['value'] = (counter*100)/total_images
            self.root.update_idletasks()
            counter += 1
        # chiusura file index
        self.progress['value'] = 0
       
        output_nome_dim.close()
        output_kp.close()
        output_des.close()
