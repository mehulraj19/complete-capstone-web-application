
from flask.globals import session
from yoga import app
from flask import render_template, redirect, url_for, flash, request, Response
from yoga.models import User
from yoga.forms import RegisterForm, LoginForm
from flask_wtf import FlaskForm
from yoga import db
from yoga import model, loaded_model, model1
from flask_login import login_user, logout_user, login_required, current_user
from wtforms import SelectField, SubmitField
from wtforms.fields import DateField
from wtforms_components import DateRange

import requests
import numpy as np
import pandas as pd
import datetime
import cv2
import os
import sys
import argparse
import ast
import cv2
import torch
import glob
import time

from statsmodels.tsa.arima.model import ARIMA as arima
from sklearn.neural_network import MLPClassifier
from vidgear.gears import CamGear
sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict

import cv2
# model = pickle.load(open('yoga_model.pkl', 'rb'))

# Variables
IMG_SIZE = 200
COUNT = 0
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,100)
fontScale              = 1
fontColor              = (0,0,0)
lineType               = 2


## FORMS FOR COVID ANALYSIS AND COVID CHECKER
class CountryForm(FlaskForm):
    countryData = SelectField('Country', choices=[])
    startdate = DateField('Start Date', format='%Y-%m-%d', default=datetime.date(2020,1,1) ,validators=[DateRange(min=datetime.date(2020,1,1), max=datetime.date.today)])
    enddate = DateField('End Date', format='%Y-%m-%d', default=datetime.date.today() ,validators=[DateRange(min=datetime.date(2020,1,1), max=datetime.date.today)])
    submit = SubmitField('Submit')

    def validate_on_submit(self):
        res = super(CountryForm, self).validate()
        if self.startdate.data > self.enddate.data:
            self.startdate.data, self.enddate.data = self.enddate.data, self.startdate.data
        return res

class CovidCheckerForm(FlaskForm):
    difficulty_in_breathing_data = SelectField('difficulty_in_breathing', choices=[], default='YES')
    diarrhea_data = SelectField('diarrhea', choices=[], default='YES')
    pains_data = SelectField('pains', choices=[], default='YES')
    fever_data = SelectField('fever', choices=[], default='YES')
    sore_throat_data = SelectField('sore_throat', choices=[], default='YES')



## WORK FOR CHARTJS FOR MOST AFFECTED COUNTRIES IN THE WORLD BY COVID
## dataset
myfile = requests.get('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
open('data', 'wb').write(myfile.content)
confirmed_global_df = pd.read_csv('data')

## pre-processing
df = confirmed_global_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='ConfirmedCases')
df1 = df.groupby(["Date", "Country/Region"])[['Date', 'Country/Region', 'ConfirmedCases']].sum().reset_index()
df1.columns = ['Date', 'Country', 'Confirmed']
df1.set_index('Date', inplace=True)

## unique countries
arr = df1.Country.unique()
main_arr = []
for country in arr:
    main_arr.append({'country': country})
country_wise_data = {}
for i in range(len(arr)):
    val = df1.loc[df1['Country']==arr[i], 'Confirmed'].sum()
    country_wise_data[arr[i]] = val
country_wise_data = sorted(country_wise_data.items(), key=lambda x: x[1])

## length/ number of countries
val = len(country_wise_data)
countries = {}
for i in range(val-1, val-11, -1):
    countries[country_wise_data[i][0]] = country_wise_data[i][1] 


# Routes

@app.route('/')
@app.route('/covidsystem')
def home_covid_system_page():
    return render_template('home_covid_system.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/leaderboard')
def leaderboard_page():
    leaders = User.query.order_by(User.score)
    return render_template('leaderboard.html', leaders=leaders)

@app.route('/choice')
@login_required
def choice_page():
    return render_template('choice.html')


@app.route('/testViaImage')
def practice_page():
    return render_template('practice.html')


## prediction via image
@app.route('/predict', methods=['GET', 'POST'])
def resultForm(filename = None, hrnet_c = 48, hrnet_j = 17, hrnet_weights = "./weights/pose_hrnet_w48_384x288.pth", hrnet_joints_set = "coco", image_resolution = '(384, 288)', single_person = True,max_batch_size = 16, disable_vidgear = False, device = None):
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    image = cv2.imread('static/{}.jpg'.format(COUNT))

    if device is not None:
            device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    model = SimpleHRNet(
        hrnet_c, 
        hrnet_j, 
        hrnet_weights,
        resolution=ast.literal_eval(image_resolution),
        multiperson = not single_person,
        max_batch_size=max_batch_size,
        device=device
    )
    no_to_label = {0:"tree", 1:"warrior1", 2:"warrior2", 3:"childs",4:"downwarddog",5:"plank",6:"mountain",7:"trianglepose"}
    image_to_blob = {}
    pts = model.predict(image)
    resolution = image.shape
    x_len = resolution[0]
    y_len = resolution[1]
    vector= []
    keypoints = pts[0]
    for pt in keypoints:
        pt = list(pt)
        temp = []
        temp.append((pt[0]/x_len))
        temp.append((pt[1]/y_len))
        vector.extend(temp)

    vector = list(vector)
    predicted_pose = model1.predict([vector])      
    data = predicted_pose[0]
    res = model1.predict_proba([vector])
    val = max(res[0])
    user = User.query.filter_by(username=current_user.username).first()
    if val > 0.95:
        user.score += 0.1
    elif val > 0.9:
        user.score += 0.05
    else:
        user.score += 0.01
    db.session.commit()
    return render_template('result.html', data=data, val=val)



@app.route('/covidanalysis', methods=['GET', 'POST'])
def covidanalysis():
    form = CountryForm()
    form.countryData.choices = [(country) for country in arr]
    # print(arr)

    if request.method == 'POST':
        country = form.countryData.data
        startDate = form.startdate.data
        endDate = form.enddate.data
        print(startDate)
        print(endDate)
                
        ## data
        myfile = requests.get('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
        open('data', 'wb').write(myfile.content)
        confirmed_global_df = pd.read_csv('data')
        ## pre-processing
        confirmed_country = confirmed_global_df[confirmed_global_df['Country/Region'] == country].reset_index(drop=True)
        df = confirmed_country.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='ConfirmedCases')
        df2 = df.groupby(["Date", "Country/Region"])[['Date', 'Country/Region', 'ConfirmedCases']].sum().reset_index()
        df2.columns = ['Date', 'Country', 'Confirmed']
        df2['Date'] = pd.to_datetime(df2['Date'])
        #sorting data
        df2.sort_values(by=['Date'], inplace=True)
        # reseting index
        df2.reset_index(drop=True, inplace=True)
        # setting up data
        confirmed = []
        confirmed.append(df2['Confirmed'][0])
        for i in range(1, len(df2)):
            confirmed.append(abs(df2['Confirmed'][i] - df2['Confirmed'][i-1]))
        df2['confirmed'] = confirmed
        df2.drop(['Confirmed'], axis=1, inplace=True)

        df2.to_csv('confirmed.csv', index=False)


        ## data
        myfile = requests.get('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
        open('data', 'wb').write(myfile.content)
        confirmed_global_df = pd.read_csv('data')
        ## pre-processing
        confirmed_country = confirmed_global_df[confirmed_global_df['Country/Region'] == country].reset_index(drop=True)
        df = confirmed_country.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='RecoveredCases')
        df2 = df.groupby(["Date", "Country/Region"])[['Date', 'Country/Region', 'RecoveredCases']].sum().reset_index()
        df2.columns = ['Date', 'Country', 'Recovered']
        df2['Date'] = pd.to_datetime(df2['Date'])
        #sorting data
        df2.sort_values(by=['Date'], inplace=True)
        # reseting index
        df2.reset_index(drop=True, inplace=True)
        # setting up data
        recovered = []
        recovered.append(df2['Recovered'][0])
        for i in range(1, len(df2)):
            recovered.append(abs(df2['Recovered'][i] - df2['Recovered'][i-1]))
        df2['recovered'] = recovered
        df2.drop(['Recovered'], axis=1, inplace=True)

        df2.to_csv('recovered.csv', index=False)


        ## data
        myfile = requests.get('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        open('data', 'wb').write(myfile.content)
        confirmed_global_df = pd.read_csv('data')
        ## pre-processing
        confirmed_country = confirmed_global_df[confirmed_global_df['Country/Region'] == country].reset_index(drop=True)
        df = confirmed_country.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='DeathCases')
        df2 = df.groupby(["Date", "Country/Region"])[['Date', 'Country/Region', 'DeathCases']].sum().reset_index()
        df2.columns = ['Date', 'Country', 'Deaths']
        df2['Date'] = pd.to_datetime(df2['Date'])
        #sorting data
        df2.sort_values(by=['Date'], inplace=True)
        # reseting index
        df2.reset_index(drop=True, inplace=True)
        # setting up data
        deaths = []
        deaths.append(df2['Deaths'][0])
        for i in range(1, len(df2)):
            deaths.append(abs(df2['Deaths'][i] - df2['Deaths'][i-1]))
        df2['deaths'] = deaths
        df2.drop(['Deaths'], axis=1, inplace=True)

        df2.to_csv('deaths.csv', index=False)

        df_confirmed = pd.read_csv('confirmed.csv')
        df_recovered = pd.read_csv('recovered.csv')
        df_deaths = pd.read_csv('deaths.csv')

        data_main = [df_confirmed['Date'], df_confirmed['confirmed'], df_recovered['recovered'], df_deaths['deaths']]
        headers = ['Date', 'confirmed', 'recovered', 'deaths']
        df_main = pd.concat(data_main, axis=1, keys=headers)
        df_main.to_csv('main.csv', index=False)


        df = pd.read_csv('main.csv')
        dates = []
        confirmed = []
        recovered = []
        deceased = []
        for i in range(len(df)):
            dates.append(df['Date'][i])
            confirmed.append(df['confirmed'][i])
            recovered.append(df['recovered'][i])
            deceased.append(df['deaths'][i])
        # print(dates)

        dates_last_seven_days = []
        confirmed_last_seven_days = []
        recovered_last_seven_days = []
        deceased_last_seven_days = []
        for i in range(len(df)-7, len(df)):
            dates_last_seven_days.append(df['Date'][i])
            confirmed_last_seven_days.append(df['confirmed'][i])
            recovered_last_seven_days.append(df['recovered'][i])
            deceased_last_seven_days.append(df['deaths'][i])



        ## Model Training
        df.set_index('Date', inplace=True)
        today = datetime.date.today()
        dates_next_seven_days = []
        for i in range(len(df)):
            dates_next_seven_days.append(today + datetime.timedelta(days=i))
        dates_next_seven_days = pd.to_datetime(dates_next_seven_days)


        model_confirmed = arima(df['confirmed'], order=(2,1,2))
        res_confirmed = model_confirmed.fit()

        model_deceased = arima(df['deaths'], order=(0,1,1))
        res_deceased = model_deceased.fit()

        model_recovered = arima(df['recovered'], order=(0,1,0))
        res_recovered = model_recovered.fit()

        y_forecast_confirmed_next_7_days = res_confirmed.forecast(steps=7)
        y_forecast_confirmed_next_7_days = y_forecast_confirmed_next_7_days.astype(np.int64)

        y_forecast_deceased_next_7_days = res_deceased.forecast(steps=7)
        y_forecast_deceased_next_7_days = y_forecast_deceased_next_7_days.astype(np.int64)

        y_forecast_recovered_next_7_days = res_recovered.forecast(steps=7)
        y_forecast_recovered_next_7_days = y_forecast_recovered_next_7_days.astype(np.int64)



        data_val = []
        for i in range(len(y_forecast_confirmed_next_7_days)):
            a = y_forecast_confirmed_next_7_days[i]
            b = y_forecast_deceased_next_7_days[i]
            c = y_forecast_recovered_next_7_days[i]
            l = []
            l.append(dates_next_seven_days[i])
            l.append(a)
            l.append(b)
            l.append(c)
            data_val.append(l)

        headers = ['Date', 'Confirmed', 'Deaths', 'Recovered']
        df_next_seven_days = pd.DataFrame(data_val, columns=headers)
        df_next_seven_days.to_csv('main_next_seven_days.csv', index=False)

        df_next_seven_days = pd.read_csv('main_next_seven_days.csv')

        dates_next_seven_days = []
        confirmed_next_seven_days = []
        recovered_next_seven_days = []
        deceased_next_seven_days = []
        for i in range(len(df_next_seven_days)):
            dates_next_seven_days.append(df_next_seven_days['Date'][i])
            confirmed_next_seven_days.append(df_next_seven_days['Confirmed'][i])
            recovered_next_seven_days.append(df_next_seven_days['Recovered'][i])
            deceased_next_seven_days.append(df_next_seven_days['Deaths'][i])


        ## range type data pre-processing
        data_values_ranged = []
        index_ranged = 0
        df_main['Date'] = pd.to_datetime(df_main['Date'])
        for i in range(len(df)):
            if(df_main['Date'][i] == startDate):
                print('Start Data Found')
                index_ranged = i
                break
        print(index_ranged)
        for i in range(index_ranged, len(df_main)):
            if(df_main['Date'][i] == endDate):
                print('End Date Found')
                break
            l = []
            l.append(df_main['Date'][i])
            l.append(df_main['confirmed'][i])
            l.append(df_main['deaths'][i])
            l.append(df_main['recovered'][i])
            data_values_ranged.append(l)
        headers_ranged = ['Date', 'confirmed', 'deaths', 'recovered']
        df_main_ranged = pd.DataFrame(data_values_ranged, columns=headers_ranged)
        df_main_ranged.to_csv('main_ranged.csv', index=False)
        
        df_main_ranged_new = pd.read_csv('main_ranged.csv')
        confirmed_ranged = []
        dates_ranged = []
        recovered_ranged = []
        deceased_ranged = []
        for i in range(len(df_main_ranged_new)):
            dates_ranged.append(df_main_ranged_new['Date'][i])
            confirmed_ranged.append(df_main_ranged_new['confirmed'][i])
            deceased_ranged.append(df_main_ranged_new['deaths'][i])
            recovered_ranged.append(df_main_ranged_new['recovered'][i])

        return render_template('covidanalysiscountries.html', country=country, dates=dates_ranged, confirmed=confirmed_ranged, recovered=recovered_ranged, deceased=deceased_ranged, dates_last_seven_days=dates_last_seven_days, confirmed_last_seven_days=confirmed_last_seven_days, recovered_last_seven_days=recovered_last_seven_days, deceased_last_seven_days=deceased_last_seven_days, dates_next_seven_days=dates_next_seven_days, confirmed_next_seven_days=confirmed_next_seven_days, recovered_next_seven_days=recovered_next_seven_days, deceased_next_seven_days=deceased_next_seven_days)
    
    label = [(country) for country in countries.keys()]
    val = [(data) for data in countries.values()]
    return render_template('countries.html', labels=label, values=val, form=form)


@app.route('/covidcheck', methods=['GET', 'POST'])
def covidcheck():
    form = CovidCheckerForm()
    main_choices = ['YES', 'NO']
    form.difficulty_in_breathing_data.choices = [(choice) for choice in main_choices]
    form.diarrhea_data.choices = [(choice) for choice in main_choices]
    form.pains_data.choices = [(choice) for choice in main_choices]
    form.fever_data.choices = [(choice) for choice in main_choices]
    form.sore_throat_data.choices = [(choice) for choice in main_choices]

    if(request.method == 'POST'):
        dib = form.difficulty_in_breathing_data.data
        dd = form.diarrhea_data.data
        pd = form.pains_data.data
        fd = form.fever_data.data
        std = form.sore_throat_data.data
        message = ''

        if dib == 'YES' and dd == 'YES' and pd == 'YES' and fd == 'YES' and std == 'YES':
            message = 'There is high chance you have Covid, Get Tested.'
        elif dib == 'YES' and dd == 'NO' and pd == 'YES' and fd == 'YES' and std == 'YES':
            message = 'There is high chance you have Covid, Get Tested.'
        elif dib == 'YES' and dd == 'NO' and pd == 'NO' and fd == 'YES' and std == 'YES':
            message = 'There is high chance you have Covid, Get Tested.'
        elif dib == 'YES' and dd == 'NO' and pd == 'NO' and fd == 'NO' and std == 'YES':
            message = 'There is less chance you have covid.'
        elif dib == 'YES' and dd == 'YES':
            message = 'There is high chance you have Covid, Get Tested.'
        elif dib == 'NO' and dd == 'YES' and pd == 'YES' and fd == 'YES' and std == 'YES':
            message = 'There is chance you have Covid, Get Tested.'
        elif dib == 'NO' and dd == 'NO' and pd == 'YES' and fd == 'YES' and std == 'YES':
            message = 'There is chance you have covid, get Tested.'
        elif dib == 'NO' and dd == 'YES' and pd == 'NO' and fd == 'YES' and std == 'YES':
            message = 'There is chance you have covid, get Tested.'
        elif dib == 'NO' and dd == 'YES' and pd == 'NO' and fd == 'NO' and std == 'YES':
            message = 'There is less chance you have covid, take rest. Avoid going out.'
        elif dib == 'NO' and dd == 'YES' and pd == 'NO' and fd == 'YES' and std == 'NO':
            message = 'There is less chance you have covid, take rest.'
        elif dib == 'NO' and dd == 'NO' and pd == 'NO' and fd == 'YES' and std == 'YES':
            message = 'There is less chance you have covid, Take Rest. Stay Indoors.'
        elif dib == 'NO' and dd == 'NO' and pd == 'NO' and fd == 'NO' and std == 'YES':
            message = 'There is very less chance you have covid!!'
        else:
            message = 'No need to worry!'
        return render_template('covidcheck.html', message=message, form=form)
    return render_template('covidcheck.html', form=form)

@app.route('/register', methods = ['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email=form.email.data,
                              password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f'Account created Successfully!! You are now logged in as {user_to_create.username}', category="success")
        return redirect(url_for('choice_page'))
    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(f'There was an error creating a user: {err_msg}', category='danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(attempted_password=form.password.data):
            login_user(attempted_user)
            flash(f'Success!! You are logged in as: {attempted_user.username}', category='success')
            return redirect(url_for('choice_page'))
        else:
            flash('Username or Password do not match, Please Try Again', category='danger')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout_page():
    logout_user()
    flash('You have been Logged Out!!', category='info')
    return redirect(url_for('home_page'))


## video camera deployment on website
ds_factor = 0.6
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
                           interpolation=cv2.INTER_AREA)
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
res = gen(VideoCamera())

@app.route('/video_feed')
def video_feed():
    return Response(res, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam')
def open_app(camera_id = 0, filename = None, hrnet_c = 48, hrnet_j = 17, hrnet_weights = "./weights/pose_hrnet_w48_384x288.pth", hrnet_joints_set = "coco", image_resolution = '(384, 288)', single_person = True,max_batch_size = 16, disable_vidgear = False, device = None):
        if device is not None:
            device = torch.device(device)
        else:
            if torch.cuda.is_available() and True:
                torch.backends.cudnn.deterministic = True
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
        image_resolution = ast.literal_eval(image_resolution)
        has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
        if filename is not None:
            video = cv2.VideoCapture(filename)
            assert video.isOpened()
        else:
            if disable_vidgear:
                video = cv2.VideoCapture(camera_id)
                assert video.isOpened()
            else:
                video = CamGear(camera_id).start()

        model = SimpleHRNet(
            hrnet_c,
            hrnet_j,
            hrnet_weights,
            resolution=image_resolution,
            multiperson=not single_person,
            max_batch_size=max_batch_size,
            device=device
        )
        
        no_to_label = {0:"tree", 1:"warrior1", 2:"warrior2", 3:"childs",4:"downwarddog",5:"plank",6:"mountain",7:"trianglepose"}
        image_to_blob = {}
        for id,path in no_to_label.items():
            images = [cv2.imread(file) for file in glob.glob('sampleposes\\'+path+'.jpg')]
            image_to_blob[id] = images
        while True:
            if filename is not None or disable_vidgear:
                ret, frame = video.read()
                if not ret:
                    break
            else:
                frame = video.read()
                if frame is None:
                    break
            pts = model.predict(frame)
            resolution = frame.shape
            x_len = resolution[0]
            y_len = resolution[1]
            vector = []
            if len(pts) == 0:
                continue
            keypoints = pts[0]

            for pt in keypoints:
                pt = list(pt)
                temp = []
                temp.append((pt[0]/x_len))
                temp.append((pt[1]/y_len))
                vector.extend(temp)

            vector = list(vector)
            predicted_pose = loaded_model.predict([vector]) 
            text = no_to_label[predicted_pose[0]] + " pose"
            cv2.putText(image_to_blob[predicted_pose[0]][0], text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType) 
            cv2.imshow("Suggestion",image_to_blob[predicted_pose[0]][0])
            k= cv2.waitKey(1)
            for i, pt in enumerate(pts):
                frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                                points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                points_palette_samples=10)

            if has_display:
                cv2.imshow('frame.png', frame)
                k = cv2.waitKey(1)
                if k == 27:  # Esc button
                    disable_vidgear=True
                    if disable_vidgear:
                        video.stop()
                        return redirect(url_for('choice_page'))
                    else:
                        video.stop()
                    break
                    
            else:
                cv2.imwrite('frame.png', frame)
