from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SelectField

main_choices = ['YES', 'NO']

class CheckerForm(FlaskForm):
    difficulty_in_breathing_data = SelectField('difficulty_in_breathing', choices=[])
    diarrhea_data = SelectField('diarrhea', choices=[])
    pains_data = SelectField('pains', choices=[])
    fever_data = SelectField('fever', choices=[])
    sore_throat_data = SelectField('sore_throat', choices=[])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'covid checker'

@app.route('/check', methods=['GET', 'POST'])
def check():
    form = CheckerForm()
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
        if dib == 'YES' and dd == 'YES' and pd == 'YES':
            message = 'There is high chance that you may have covid'
        elif dib == 'YES' and dd == 'YES':
            message = 'There is high chance that you may have covid'
        elif dib == 'YES' and pd == 'YES':
            message = 'There is high chance that you may have covid'
        elif dib == 'NO' and dd == 'NO' and pd == 'NO' and fd == 'YES' and std == 'YES':
            message = 'There is less chance that you have covid, still it\'s advisable to have covid test'
        elif dib == 'NO' and dd == 'NO' and pd == 'NO':
            message = 'There is less chance that you may have covid so need to worry, take rest!!'
        elif dib == 'NO' and dd == 'NO' and pd == 'YES':
            message = 'There is less chance that you have covid, take Paracetamol!!'
        elif dib == 'NO' and dd == 'YES' and pd == 'YES' and fd == 'YES':
            message = ' it\'s advisable to have covid test'
        elif dib == 'NO' and dd == 'YES' and pd == 'YES' and fd == 'YES' and std == 'YES':
            message = ' it\'s advisable to have covid test'
        else:
            message = 'No need to worry!!'
        return render_template('page.html', message=message, form=form)

    return render_template('page.html', form=form)