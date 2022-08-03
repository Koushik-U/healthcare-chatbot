from flask import Flask,render_template,request,redirect,url_for,session
import tflearn
import numpy as np
import pandas as pd
import pickle,random
import json
import sqlite3
import nltk
import MySQLdb
from nltk.stem.lancaster import LancasterStemmer
import disease_predictors as dp
from datetime import date
stemmer = LancasterStemmer()
mydb = MySQLdb.connect(host='localhost',user='root',passwd='root',db='healthcarebot')
conn = mydb.cursor()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

with open("assets/input_data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

with open("assets/intents.json") as myfile:
        data = json.load(myfile)

#tf.reset_default_graph()

network = tflearn.input_data(shape=[None, len(training[0])])

network = tflearn.fully_connected(network,8)
network = tflearn.fully_connected(network,8)

network = tflearn.fully_connected(network,len(output[0]),activation="softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)

model.load("assets/chatbot.tflearn")

chats=[]
@app.route("/") #home
def hello():
        return render_template("home.html")

@app.route('/logon')
def logon():
        return render_template('signup.html')

@app.route('/login')
def login():
        return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("form.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
            session['uname'] = mail1
            return render_template("form.html")
    else:
        return render_template("signup.html")
@app.route('/history')
def history():
        uname=session.get('uname')
        cmd="SELECT * FROM history WHERE uname='"+str(uname)+"'"
        conn.execute(cmd)
        data=conn.fetchall()
        return render_template("his.html",results=data)
        
@app.route('/form')
def form():
    return render_template('form.html')
def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele+","  
    
    # return string  
    return str1 
@app.route("/start1",methods=['POST','GET'])
def start1():
        multiselect = request.form.getlist('symp')
        symp=listToString(multiselect)
        uname=session.get('uname')
        dname,did=dp.process(multiselect)
        print("Predicted did and name",did,dname)
        medicine=pd.read_csv("disease_medicine.csv",encoding='latin1')
        suitablemedicinename=medicine.loc[medicine['DID'] == did]['Medicine_Name'].values
        suitableMedicine_Composition=medicine.loc[medicine['DID'] == did]['Medicine_Composition'].values
        suitableMedicine_Description=medicine.loc[medicine['DID'] == did]['Medicine_Description'].values
        medicine_recommend="Medicine Name: "+suitablemedicinename+"\n  Medicine Composition: "+suitableMedicine_Composition+"\n Medicine Description: "+suitableMedicine_Description
        precation=pd.read_csv("disease_precaution.csv",encoding='latin1')
        suitable_precation1=precation.loc[precation['Disease'] == dname]['Precaution_1'].values
        suitable_precation2=precation.loc[precation['Disease'] == dname]['Precaution_2'].values
        suitable_precation3=precation.loc[precation['Disease'] == dname]['Precaution_3'].values
        suitable_precation4=precation.loc[precation['Disease'] == dname]['Precaution_4'].values
        suitable_precation=suitable_precation1+" "+suitable_precation2+" "+suitable_precation3+" "+suitable_precation4
        riskfactor=pd.read_csv("disease_riskFactors.csv",encoding='latin1')
        suitable_PRECAU=riskfactor.loc[riskfactor['DID'] == did]['PRECAU'].values
        suitable_OCCUR=riskfactor.loc[riskfactor['DID'] == did]['OCCUR'].values
        suitable_RISKFAC=riskfactor.loc[riskfactor['DID'] == did]['RISKFAC'].values
        suitablemedicinename=listToString(suitablemedicinename)
        doctor=pd.read_csv("doctor.csv")
        doctorname=doctor.loc[doctor['disease_id']==did]['doctor_name'].values
        doctorname=listToString(doctorname)
        doctorname=doctorname.replace("'","")
        doctorname=doctorname.replace("]","")
        cur_date=date.today()
        
        cmd="INSERT INTO history(uname,symp,predic_dis,med,doc,dat) Values('"+str(uname)+"','"+str(symp)+"','"+str(dname)+"','"+str(suitablemedicinename)+"','"+str(doctorname)+"','"+str(cur_date)+"')"
        print(cmd)
        print("Inserted Successfully")
        conn.execute(cmd)
        mydb.commit()
        chats.append("Doctor Name: "+str(doctorname))
        chats.append("Riskfactor:"+str(suitable_RISKFAC))
        chats.append("Spreading Type: "+str(suitable_OCCUR))
        chats.append("Precations: "+str(suitable_precation)+str(suitable_PRECAU))
        chats.append("Medicine Recommendation "+str(medicine_recommend))
        chats.append("Predicted Disease"+str(dname))
        chats.append("You: "+str(multiselect))
        
        
        
        
        
        
        return render_template('form.html',chats=chats[::-1],type="")



@app.route("/start",methods=['POST','GET'])
def start():
        inp = [str(x) for x in request.form.values()]
        print(inp[0])
        #return render_template('chat_bot.html',result=inp[0])
        if inp[0]=="Hi" or inp[0]=="hi" or inp[0]=="Hai" or inp[0]=="hai":
                result="Hai welcome to Chat bot do you want Help"
                chats.append(result)
                return render_template('form.html',chats=chats[::-1],type="")
        if inp[0]=="Yes" or inp[0]=="yes":
                return render_template('form1.html',chats=chats[::-1],type="")
        if inp[0]=="No" or inp[0]=="no":
                results = model.predict([bag_of_words(inp[0],words)])[0]
                print(results)
                results_index = np.argmax(results)
                tag = labels[results_index]
                print(tag)  
                if results[results_index] < 0.8 or len(inp[0])<2:
                        result ="Sorry, I didn't get you. Please try again."
                        chats.append(result)
                        #return render_template('form.html',chats=chats[::-1],type="")
                else:
                        for tg in data['intents']:
                                if tg['tag'] == tag:
                                        responses = tg['responses']
                                        result=""+random.choice(responses)
                                        chats.append("You: " + inp[0])
                                        chats.append(result)
                if result=="Sorry, I didn't get you. Please try again.":
                        print("Machine not understood")
                        chats.append(result)
                        return render_template('form.html',chats=chats[::-1],type="")
                else:
                        return render_template('form.html',chats=chats[::-1],type="")
                return render_template('form.html',chats=chats[::-1],type="")
               
        else:
                results = model.predict([bag_of_words(inp[0],words)])[0]
                print(results)
                results_index = np.argmax(results)
                tag = labels[results_index]
                print(tag)  
                if results[results_index] < 0.8 or len(inp[0])<2:
                        result ="Sorry, I didn't get you. Please try again."
                else:
                        for tg in data['intents']:
                                if tg['tag'] == tag:
                                        responses = tg['responses']
                                        result=""+random.choice(responses)
                                        chats.append("You: " + inp[0])
                                        chats.append(result)
                if result=="Sorry, I didn't get you. Please try again.":
                        print("Machine not understood")
                        chats.append(result)
                        return render_template('form.html',chats=chats[::-1],type="")
                else:
                        return render_template('form.html',chats=chats[::-1],type="")
        return render_template('form.html',chats=chats[::-1],type="")



        
def bag_of_words(s,words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
                for i,w in enumerate(words):
                        if w == se:
                                bag[i] = 1

        return np.array(bag)

                        
# start() 
if __name__=="__main__":
        app.run(debug=True)

