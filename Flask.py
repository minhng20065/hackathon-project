pip install Flask
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/action_page.php')
def index():
    return render_template('Untitled-1.html')  # Render the HTML form

@app.route('/action_page.php', methods=['POST'])
def process():
    # Retrieve user input from the form
    email = request.form['semail']
    
def cut(email):
    return email.split("@", 1)[1]

# as of right now this pipeline should only take the user input for the "sender email" and
# it shoudl output the second half of the email address after the "@" sign.
# (ex. "test@gmail.com" -> "gmail.com"). This will not be the actual output for our project,
# its just a placeholder and a test to see if the HTML Input -> Flask -> Python script
# pathway works. In theory, the link "/action_page.php" should be present in both the
# HTML and the Flask/python code, which is how the front and back ends are linked. 
# In order to adapt it to our final python/ML script, we will just have to change
# the last function ("cut") to whatever functions we are using in our final project! ^_^
