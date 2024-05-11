import csv
import os
from process_image import image_data_extract

def get_data_from_folder( data:list, export_file_name:str ):

    list = []

    for path in data:
        hand = path[-5]
        finger_count = int(path[-6])

        list.append( image_data_extract(path, hand, finger_count) )
    
    # Example: Writing data to a CSV file
    with open(export_file_name, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow( ['path', 'rightHand', 'fingersCount', 'density', 'angle_x', 'angle_y', 'meanX', 'meanY', 'mean_middle_point'] )
        
        for row in list:
            writer.writerow( row )


def main():

    trainpath = os.listdir("../../train")
    testpath = os.listdir("../../test")

    traindata = ['../../train/' + i for i in trainpath]  
    testdata = ["../../test/" + i for i in testpath]  

    print("Processando dados de treino")
    get_data_from_folder( traindata, "train-extract.csv" )

    print("Processando dados de teste")
    get_data_from_folder( testdata, "test-extract.csv" )
    

main()
