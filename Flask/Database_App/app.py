from flask import Flask, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy

# Initialize
app = Flask(__name__)

# DB
db = SQLAlchemy(app)
# Database path in local directory
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/database/users.db'

# DB model schema
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(50))
    lastname = db.Column(db.String(50))

# Now to create the table in the database 'users.db' in the above created and specified directory '/static/database/':
# Open Terminal (from within VSCode possible) and go to project directory
# Open Python (venv) ...>python
# >>> from app import db
# >>> db.create_all()
# >>> exit()

# Route
@app.route('/')
def index():
	return "Hello ML and data science Apps!"

# Adding HTML
@app.route('/home')
def home():
	return render_template('home.html')

# Writing submitted form entries to local database
@app.route('/predict', methods=['GET', 'POST'])
def predict():    
    if request.method=='POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        single_user = User(firstname=firstname, lastname=lastname)
        db.session.add(single_user)
        db.session.commit()
    return render_template('home.html', firstname=firstname.upper(), lastname=lastname.upper())

# Retrieving users data from database
@app.route('/allusers')
def allusers():    
    userslist = User.query.all()
    return render_template('results.html', userslist=userslist)

# Searching user data from database via Dynamic URL Query
@app.route('/profile/<firstname>')
def profile(firstname):
    # Searching user data from database
    user = User.query.filter_by(firstname=firstname).first()
    print(user)
    return render_template('profile.html', user=user)

# Templating
@app.route('/about')
def about():
    message = "Creating ML and data science apps with Python"
    return render_template('about.html', message=message)

# Set debug=False for production
if __name__ == '__main__':
	app.run(debug=True)