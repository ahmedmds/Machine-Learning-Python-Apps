import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
import datetime
from PIL import Image

# To hide hamburger (top right corner) and “Made with Streamlit” footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Get encoding (value) from mapping dictionary
def get_value (val, mapping_dict):
    for key, value in mapping_dict.items():
        if val == key:
            return value

# Get string (key) from mapping dictionary
def get_key (val, mapping_dict):
    for key, value in mapping_dict.items():
        if val == value:
            return key

# Load model
def load_prediction_model (model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

class Monitor(object):

    conn = sqlite3.connect('data.db')
    cur = conn.cursor()

    # def __init__(self, **kwargs): # Alternative
    def __init__(self, age=None ,workclass=None ,fnlwgt=None ,education=None ,education_num=None , \
                    marital_status=None ,occupation=None ,relationship=None ,race=None ,sex=None , \
                        capital_gain=None ,capital_loss=None ,hours_per_week=None ,native_country=None, \
                            predicted_class=None,model_class=None, time_of_prediction=None):      
        super (Monitor, self).__init__()
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country
        self.predicted_class = predicted_class
        self.model_class = model_class
        self.time_of_prediction = time_of_prediction

    def __repr__(self):
        # return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},native_country ={self.native_country},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
        "Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt}, education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},native_country = {self.native_country},predicted_class = {self.predicted_class},model_class = {self.model_class}, time_of_prediction={self.time_of_prediction})".format(self=self)

    def create_table(self):
        self.cur.execute('CREATE TABLE IF NOT EXISTS predictiontable(age NUMERIC,workclass NUMERIC,fnlwgt NUMERIC,education NUMERIC,education_num NUMERIC,marital_status NUMERIC,occupation NUMERIC,relationship NUMERIC,race NUMERIC,sex NUMERIC,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country NUMERIC,predicted_class NUMERIC,model_class TEXT, time_of_prediction TEXT)')

    def add_data(self):
        self.cur.execute('INSERT INTO predictiontable(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,predicted_class,model_class,time_of_prediction) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class, self.time_of_prediction))
        self.conn.commit()

    def view_all_data(self):
        self.cur.execute('SELECT * FROM predictiontable')
        data = self.cur.fetchall()
        # for row in data:
        # 	print(row)
        return data

def main():
    """ Salary Predictor with ML"""

    st.title("Salary Predictor")
    section_list = ["Exploratory Data Analysis (EDA)", "EDA by Country", "Prediction", "Metrics"]
    section = st.sidebar.selectbox("Choose an activity", section_list)

    # Load data
    df = pd.read_csv("data/adult_salary.csv", index_col = 0)
    
    # Section heading
    st.subheader(section)

    ########## EDA Section ##########
    if section == "Exploratory Data Analysis (EDA)":
        # Preview data
        if st.checkbox("Preview Dataset"):
            number = st.number_input("Number of rows to show in head", min_value=1, max_value=10, value=5, step=1)  # 'value' param is default value      
            st.dataframe(df.head(number))

        # Show columns & rows
        if st.checkbox("Column names"):
            st.write(df.columns)
        
        # Description
        if st.checkbox("Show description"):
            st.write(df.describe())

        # Shape
        if st.checkbox("Show shape of dataset"):
            st.write(df.shape)
            data_dim = st.radio("Show dimensions by", ("Rows", "Columns"))
            if data_dim == "Rows":
                st.text("Number of rows:")
                st.write(df.shape[0])
            elif data_dim == "Columns":
                st.text("Number of columns:")
                st.write(df.shape[1])           

        # Select particular columns/rows
        if st.checkbox("Select columns to show"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select columns", all_columns)
            if len(selected_columns)>0:
                new_df = df[selected_columns]
                st.dataframe(new_df)

        if st.checkbox("Select row indices to show"):
            selected_indices = st.multiselect("Select row indices", df.head(10).index)
            if len(selected_indices)>0:
                new_df = df.loc[selected_indices]
                st.dataframe(new_df)

        # Value counts by column

        if st.checkbox("Value counts by a column"):
            # column_name = st.text_input("Column name", "class") # User types input, "class" is default input
            column_name = st.selectbox("Select a column", ['Select a column']+df.columns.tolist(), 0)
            if column_name in df.columns.tolist():
                st.write(df[column_name].value_counts().reset_index(level=0).rename({'index': column_name, column_name: 'Count'}, axis=1))

        # Correlation plots
        if st.checkbox("Show correlation plot (Matplotlib)"):
            plt.matshow(df.corr())
            st.pyplot(plt)

        if st.checkbox("Show correlation plot (Seaborn)"):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot(plt)


    ########## Countries Section ##########
    elif section == "EDA by Country":
        
        # List of countries
        d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
        selected_country = st.selectbox("Select a country", tuple(d_native_country.keys()))

        # Selection country
        st.text(selected_country)
        
        result_df = df[df['native-country']==get_value(selected_country, d_native_country)]
        st.dataframe(result_df)

        countries_images = {'af': 'Afghanistan','al': 'Albania','dz': 'Algeria','as': 'American Samoa','ad': 'Andorra','ao': 'Angola','ai': 'Anguilla','aq': 'Antarctica','ag': 'Antigua And Barbuda','ar': 'Argentina','am': 'Armenia','aw': 'Aruba','au': 'Australia','at': 'Austria','az': 'Azerbaijan','bs': 'Bahamas','bh': 'Bahrain','bd': 'Bangladesh','bb': 'Barbados','by': 'Belarus','be': 'Belgium','bz': 'Belize','bj': 'Benin','bm': 'Bermuda','bt': 'Bhutan','bo': 'Olivia','ba': 'Bosnia And Herzegovina','bw': 'Botswana','bv': 'Bouvet Island','br': 'Brazil','io': 'British Indian Ocean Territory','bn': 'Brunei Darussalam','bg': 'Bulgaria','bf': 'Burkina Faso','bi': 'Burundi','kh': 'Cambodia','cm': 'Cameroon','ca': 'Canada','cv': 'Cape Verde','ky': 'Cayman Islands','cf': 'Central African Republic','td': 'Chad','cl': 'Chile','cn': "People'S Republic Of China",'cx': 'Hristmas Island','cc': 'Cocos (Keeling) Islands','co': 'Colombia','km': 'Comoros','cg': 'Congo','cd': 'Congo, The Democratic Republic Of','ck': 'Cook Islands','cr': 'Costa Rica','ci': "Côte D'Ivoire",'hr': 'Croatia','cu': 'Cuba','cy': 'Cyprus','cz': 'Czech Republic','dk': 'Denmark','dj': 'Djibouti','dm': 'Dominica','do': 'Dominican Republic','ec': 'Ecuador','eg': 'Egypt','eh': 'Western Sahara','sv': 'El Salvador','gq': 'Equatorial Guinea','er': 'Eritrea','ee': 'Estonia','et': 'Ethiopia','fk': 'Falkland Islands (Malvinas)','fo': 'Aroe Islands','fj': 'Fiji','fi': 'Finland','fr': 'France','gf': 'French Guiana','pf': 'French Polynesia','tf': 'French Southern Territories','ga': 'Gabon','gm': 'Gambia','ge': 'Georgia','de': 'Germany','gh': 'Ghana','gi': 'Gibraltar','gr': 'Greece','gl': 'Greenland','gd': 'Grenada','gp': 'Guadeloupe','gu': 'Guam','gt': 'Guatemala','gn': 'Guinea','gw': 'Guinea-Bissau','gy': 'Guyana','ht': 'Haiti','hm': 'Heard Island And Mcdonald Islands','hn': 'Honduras','hk': 'Hong Kong','hu': 'Hungary','is': 'Iceland','in': 'India','id': 'Indonesia','ir': 'Iran, Islamic Republic Of','iq': 'Iraq','ie': 'Ireland','il': 'Israel','it': 'Italy','jm': 'Jamaica','jp': 'Japan','jo': 'Jordan','kz': 'Kazakhstan','ke': 'Kenya','ki': 'Kiribati','kp': "Korea, Democratic People'S Republic Of",'kr': 'Korea, Republic Of','kw': 'Kuwait','kg': 'Kyrgyzstan','la': "Lao People'S Democratic Republic",'lv': 'Latvia','lb': 'Lebanon','ls': 'Lesotho','lr': 'Liberia','ly': 'Libyan Arab Jamahiriya','li': 'Liechtenstein','lt': 'Lithuania','lu': 'Luxembourg','mo': 'Macao','mk': 'Macedonia, The Former Yugoslav Republic Of','mg': 'Madagascar','mw': 'Malawi','my': 'Malaysia','mv': 'Maldives','ml': 'Mali','mt': 'Malta','mh': 'Marshall Islands','mq': 'Martinique','mr': 'Mauritania','mu': 'Mauritius','yt': 'Mayotte','mx': 'Mexico','fm': 'Micronesia, Federated States Of','md': 'Moldova, Republic Of','mc': 'Monaco','mn': 'Mongolia','ms': 'Montserrat','ma': 'Morocco','mz': 'Mozambique','mm': 'Myanmar','na': 'Namibia','nr': 'Nauru','np': 'Nepal','nl': 'Netherlands','an': 'Netherlands Antilles','nc': 'New Caledonia','nz': 'New Zealand','ni': 'Nicaragua','ne': 'Niger','ng': 'Nigeria','nu': 'Niue','nf': 'Norfolk Island','mp': 'Northern Mariana Islands','no': 'Norway','om': 'Oman','pk': 'Pakistan','pw': 'Palau','ps': 'Palestinian Territory, Occupied','pa': 'Panama','pg': 'Papua New Guinea','py': 'Paraguay','pe': 'Peru','ph': 'Philippines','pn': 'Pitcairn','pl': 'Poland','pt': 'Portugal','pr': 'Puerto Rico','qa': 'Qatar','re': 'Réunion','ro': 'Romania','ru': 'Russian Federation','rw': 'Rwanda','sh': 'Saint Helena','kn': 'Saint Kitts And Nevis','lc': 'Saint Lucia','pm': 'Saint Pierre And Miquelon','vc': 'Saint Vincent And The Grenadines','ws': 'Samoa','sm': 'San Marino','st': 'Sao Tome And Principe','sa': 'Saudi Arabia','sn': 'Senegal','cs': 'Serbia And Montenegro','sc': 'Seychelles','sl': 'Sierra Leone','sg': 'Singapore','sk': 'Slovakia','si': 'Slovenia','sb': 'Solomon Islands','so': 'Somalia','za': 'South Africa','gs': 'South Georgia And South Sandwich Islands','es': 'Spain','lk': 'Sri Lanka','sd': 'Sudan','sr': 'Suriname','sj': 'Svalbard And Jan Mayen','sz': 'Swaziland','se': 'Sweden','ch': 'Switzerland','sy': 'Syrian Arab Republic','tw': 'Taiwan, Republic Of China','tj': 'Tajikistan','tz': 'Tanzania, United Republic Of','th': 'Thailand','tl': 'Timor-Leste','tg': 'Togo','tk': 'Tokelau','to': 'Tonga','tt': 'Trinidad And Tobago','tn': 'Tunisia','tr': 'Turkey','tm': 'Turkmenistan','tc': 'Turks And Caicos Islands','tv': 'Tuvalu','ug': 'Uganda','ua': 'Ukraine','ae': 'United Arab Emirates','gb': 'United Kingdom','us': 'United States','um': 'United States Minor Outlying Islands','uy': 'Uruguay','uz': 'Uzbekistan','ve': 'Venezuela','vu': 'Vanuatu','vn': 'Viet Nam','vg': 'British Virgin Islands','vi': 'U.S. Virgin Islands','wf': 'Wallis And Futuna','ye': 'Yemen','zw': 'Zimbabwe'}
        for k,v in countries_images.items():
            if v==selected_country:
                img_name = 'cflags/{}.png'.format(k)
                img = Image.open(os.path.join(img_name))
                st.image(img)
                
        if st.checkbox("Select columns to show"):
            result_df_columns = result_df.columns.tolist()
            selected_columns_country = st.multiselect("Select column(s)", result_df_columns)
            df_selected_columns_country = df[selected_columns_country]
            if len(df_selected_columns_country)>0:
                st.dataframe(df_selected_columns_country)
                if st.button("Plot country"):
                    st.area_chart(df_selected_columns_country)


    ########## Prediction Section ##########
    elif section == "Prediction":
        
        d_workclass = {"Never-worked": 0, "Private": 1, "Federal-gov": 2, "?": 3, "Self-emp-inc": 4, "State-gov": 5, "Local-gov": 6, "Without-pay": 7, "Self-emp-not-inc": 8}

        d_education = {"Some-college": 0, "10th": 1, "Doctorate": 2, "1st-4th": 3, "12th": 4, "Masters": 5, "5th-6th": 6, "9th": 7, "Preschool": 8, "HS-grad": 9, "Assoc-acdm": 10, "Bachelors": 11, "Prof-school": 12, "Assoc-voc": 13, "11th": 14, "7th-8th": 15}

        d_marital_status = {"Separated": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 2, "Married-civ-spouse": 3, "Never-married": 4, "Widowed": 5, "Divorced": 6}

        d_occupation = {"Tech-support": 0, "Farming-fishing": 1, "Prof-specialty": 2, "Sales": 3, "?": 4, "Transport-moving": 5, "Armed-Forces": 6, "Other-service": 7, "Handlers-cleaners": 8, "Exec-managerial": 9, "Adm-clerical": 10, "Craft-repair": 11, "Machine-op-inspct": 12, "Protective-serv": 13, "Priv-house-serv": 14}

        d_relationship = {"Other-relative": 0, "Not-in-family": 1, "Own-child": 2, "Wife": 3, "Husband": 4, "Unmarried": 5}

        d_race = {"Amer-Indian-Eskimo": 0, "Black": 1, "White": 2, "Asian-Pac-Islander": 3, "Other": 4}

        d_sex = {"Female": 0, "Male": 1}

        d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}

        d_class = {">50K": 0, "<=50K": 1}

        # User input
        age = st.slider("Select Age", min(df['age']), max(df['age']))
        workclass = st.selectbox("Select Work Class", tuple([get_key(encoded_val, d_workclass) for encoded_val in df['workclass'].unique()]))
        fnlwgt = st.number_input("Select FNLWGT", min_value = float(min(df['fnlwgt'])), max_value = float(max(df['fnlwgt'])), step=0.01)
        education = st.selectbox("Select Education", tuple([get_key(encoded_val, d_education) for encoded_val in df['education'].unique()]))
        education_num = st.slider("Select Education Level (num)", min(df['education-num']), max(df['education-num']))
        marital_status = st.selectbox("Select Marital Status", tuple([get_key(encoded_val, d_marital_status) \
                                            for encoded_val in df['marital-status'].unique()]))
        occupation = st.selectbox("Select Occupation", tuple([get_key(encoded_val, d_occupation) \
                                            for encoded_val in df['occupation'].unique()]))
        relationship = st.selectbox("Select Relationship", tuple([get_key(encoded_val, d_relationship) \
                                            for encoded_val in df['relationship'].unique()]))
        race = st.selectbox("Select Race", tuple([get_key(encoded_val, d_race) \
                                            for encoded_val in df['race'].unique()]))
        sex = st.radio("Select Sex", tuple([get_key(encoded_val, d_sex) \
                                            for encoded_val in df['sex'].unique()]))                                            
        capital_gain = st.number_input("Select Capital Gain", min_value = min(df['capital-gain']), max_value = max(df['capital-gain']))
        capital_loss = st.number_input("Select Capital Loss", min_value = min(df['capital-loss']), max_value = max(df['capital-loss']))
        hours_per_week = st.number_input("Select Hours per Week", min_value = min(df['hours-per-week']), \
                                                                                max_value = max(df['hours-per-week']))
        native_country = st.selectbox("Select Native Country", tuple([get_key(encoded_val, d_native_country) \
                                    for encoded_val in df['native-country'].unique()]))

        # Converting user inputs to encoded value
        k_workclass = get_value(workclass, d_workclass)
        k_education = get_value(education, d_education)
        k_marital_status = get_value(marital_status, d_marital_status)
        k_occupation = get_value(occupation, d_occupation)
        k_relationship = get_value(relationship, d_relationship)
        k_race = get_value(race, d_race)
        k_sex = get_value(sex, d_sex)
        k_native_country = get_value(native_country, d_native_country)

        # Results of user input
        selected_options = [age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, \
                                    race, sex, capital_gain, capital_loss, hours_per_week, native_country]
        vectorized_results = [age, k_workclass, fnlwgt, k_education, education_num, k_marital_status, k_occupation, k_relationship, \
                                    k_race, k_sex, capital_gain, capital_loss, hours_per_week, k_native_country]
        vectorized_results_array = np.array(vectorized_results).reshape(1, -1)
        st.info(selected_options)
        prettified_result = {"age": age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education, 'education_num': education_num, \
                                    'marital_status': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, \
                                    'sex': sex, 'capital_gain': capital_gain, 'capital_loss': capital_loss, 'hours_per_week': hours_per_week, \
                                        'native_country': native_country}
        st.json(prettified_result)

        st.write(vectorized_results)

        # Making predictions
        st.subheader("Prediction")
        if st.checkbox("Make Prediction"):
            all_ml_models_list = ["Logistic Regression", "Random Forest", "Naive Bayes"]
            
            # Model selection
            model_choice = st.selectbox("Model choice", all_ml_models_list)

            if st.button("Predict"):
                if model_choice == "Logistic Regression":
                    model = load_prediction_model("models/salary_logit_model.pkl")
                elif model_choice == "Random Forest":
                    model = load_prediction_model("models/salary_rf_model.pkl")           
                elif model_choice == "Naive Bayes":
                    model = load_prediction_model("models/salary_nv_model.pkl")
                
                prediction_encod = model.predict(vectorized_results_array)
                prediction_labels_dict = d_class
                prediction_str = get_key(prediction_encod, prediction_labels_dict)

                time_of_prediction = datetime.datetime.now()
                monitor = Monitor(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,prediction_str,model_choice,time_of_prediction)
                monitor.create_table()
                monitor.add_data()

                st.success('Predicted salary as: {}'.format(prediction_str))
            

    ########## Metrics Section ##########
    elif section == "Metrics":
        cnx = sqlite3.connect('data.db')
        mdf = pd.read_sql_query("SELECT * FROM predictiontable", cnx)
        st.dataframe(mdf)



if __name__ == '__main__':
    main()