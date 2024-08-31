

To manage dependencies, we are using a virtual environment. I have created it named "venv". 
* Activate the venv every time you open a terminal
* Update the requirements.txt file every time you install a package
* Use the requirements.txt file every time you set up the project.  

1. Activate the venv by typing the following commands into the terminal (don't include quotations)
    * Windows: ```.\venv\Scripts\activate```
    * macOS and Linux: ```source venv/bin/activate```

    Ensure the venv is activated whenever you are running commands on the terminal  

2. Update the requirements file every time you install a package.
    * Run the command to install the package
    * Run the command ```pip freeze > requirements.txt```  

    I have installed pandas, here is what I did  
    ```pip install pandas```  
    ```pip freeze > requirements.txt```

    As you can see, pandas is in the requirements.txt file (along with 5 other packages it installed with it)  

3. Use the requriements.txt file when setting up the project
  *  Simply run "pip install -r requirements.txt" to install all of the appropriate packages

Troubleshooting: If running into errors with dependencies, try the following command in your terminal with the virtual environment (venv) activated:

pip install --upgrade Flask Werkzeug numpy scikit-learn matplotlib



