from django.shortcuts import render
# These libraries are used by Django for rendering your pages.

from django.http import HttpResponse

from django.shortcuts import render, redirect

import numpy as np

import joblib

loaded_rf_model = joblib.load("E:/ML/rent_prj/houserent/ML_Model/rf_model.joblib")

def index(request):
    if request.method == 'POST':
        nobhk=request.POST.get('nobhk','default')
        housesize=request.POST.get('housesize','default')
        Areatype=request.POST.get('Areatype','default')
        Pincode=request.POST.get('Pincode','default')
        FurnishStatus=request.POST.get('FurnishStatus','default')
        TenantType=request.POST.get('TenantType','default')
        noofbath=request.POST.get('noofbath','default')
        features = np.array([[int(nobhk),int(housesize),int(Areatype),int(Pincode),int(FurnishStatus),int(TenantType),int(noofbath)]])
        our_labels = loaded_rf_model.predict(features)
        details={"answer":our_labels[0]}
        return render(request,"results.html",details)
    return render(request,"index.html")
# Create your views here.
