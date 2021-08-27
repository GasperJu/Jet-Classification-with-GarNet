import numpy as np
import pandas as pd

def data_adaption_JetNet(particles_number_per_jet = None, data = None):
    particles = np.arange(particles_number_per_jet)
    header = []
    etar = []
    phir = []
    ptr = []
    maskh = []
    for e, p in enumerate(particles):
        header.append('eta relative '+str(particles[p]))
        etar.append('eta relative '+str(particles[p]))
        header.append('phi relative '+str(particles[p]))
        phir.append('phi relative '+str(particles[p]))
        header.append('pt relative '+str(particles[p]))
        ptr.append('pt relative '+str(particles[p]))
        header.append('mask '+str(particles[p]))
        maskh.append('mask '+str(particles[p]))
    dt = pd.read_csv( data, header = None)
    dt = dt[0].str.split(' ', expand=True)
    dt.columns = header
    dt = dt.drop(columns=maskh)
    return dt

def lable_types(dtall, class_lable):
    lableax = dtall.pop('Type')
    lable = pd.DataFrame(columns=(class_lable)) 
    for i in range(len(class_lable)):
        lable[class_lable[i]] = (lableax == i+1)*(1.0)
    return(lable)

def cut_groups_data(dtall, lable):
    
    #data cuts
    cut_train = int(len(dtall)/3)
    cut_train2 = 2*int(len(dtall)/3) + int((len(dtall)/3)/2)
    
    #train group - bigger group
    inputTrain = dtall[:2*cut_train] 
    lableTrain = lable[:2*cut_train].values
    
    #validation group
    inputVal = dtall[2*cut_train:cut_train2]
    lableVal = lable[2*cut_train:cut_train2].values
    
    #test group (used in prediction)
    inputTest = dtall[cut_train2:]
    lableTest = lable[cut_train2:].values
    
    return(inputTrain, lableTrain, inputVal, lableVal, inputTest, lableTest)

def percentage_info(preds = None, lableTest = None):
    #counters
    top = 0 
    gluon = 0
    errtop = 0 
    errgluon = 0
    for i in range(preds.shape[0]):
        #how many tops and gluon are there?
        if lableTest[i].argmax() == preds[i].argmax():
            if lableTest[i].argmax() == 0:
                gluon = (gluon + 1)
            else:
                top = (top + 1)
        else: #how many false positives are there?
            if preds[i].argmax() == 0:
                errgluon = (errgluon + 1)
            else:
                errtop = (errtop + 1)
    return(top, gluon, errgluon, errtop)
