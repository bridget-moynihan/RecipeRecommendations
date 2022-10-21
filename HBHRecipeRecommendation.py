
######## Halfbaked Harvest recipe recommendation

#import necessary libraries

import requests
from bs4 import BeautifulSoup
import os.path
from os import path
import time
from datetime import timedelta
from datetime import date
from dateutil import parser

import pandas as pd


#Libraries for preprocessing
from gensim.parsing.preprocessing import remove_stopwords
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#Libraries for vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

#Libraries for clustering
from sklearn.cluster import KMeans



#  The way the HBH website is set up, there are 12 recipe cards per page, so 
# this function gets all recipe links on a certain page
# for each link, it calls getRecipeDetails()
# the function returns a data frame of all recipes with their information
def getRecipes(pageNumber):
    d = {'Link':[], 'Recipe': [], 'Course': [], 'Cuisine': [], 'Key Ingredients': [], 'All Ingredients':[]}
    RECIPES = pd.DataFrame(data=d)
    for i in range(pageNumber):
        url = "https://www.halfbakedharvest.com/category/recipes/page/" + str(i) + "/"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        navigation = soup.find_all("div", class_="post-summary__image")
        for item in navigation:
            recipeLink = item.find("a", href=True)['href']
            recipeDetails = getRecipeDetails(recipeLink)
            if recipeDetails != [None,None,None,None]:
                RECIPES.loc[len(RECIPES)] = recipeDetails
    return RECIPES     
            
# function for getting recipe details 
# function scrapes recipe card and grabs certain details 
# function returns a list of recipe details
def getRecipeDetails(recipeUrl):
    page = requests.get(recipeUrl)
    soup = BeautifulSoup(page.content, "html.parser")
    navigation = soup.find_all("p", class_="page-header__recipe-skip-link")
    recipeNumber = navigation[0].find("a")
    
    # sometimes HBH posts blogs that come up in the recipe index, in this case,
    # they dont have a recipe number, so I return a list of None if its a blog post
    try:
        recipeNumber = recipeNumber['data-recipe']
    except:
        return [None,None,None,None]
    
    
    printURL = "https://www.halfbakedharvest.com/wprm_print/" + recipeNumber + "/"
    printPage = requests.get(printURL)
    soup = BeautifulSoup(printPage.content, "lxml")
    
    # type of course (ex: Main course)
    try:
        course = str((soup.find("span", class_="wprm-recipe-course wprm-block-text-normal")).getText())
    except:
        course = ""
    
    # type of cusine (Ex: American)
    try:
        cuisine = (soup.find("span", class_="wprm-recipe-cuisine wprm-block-text-normal")).getText()
    except:
        cuisine = ""
    # key ingredients (this field has only been updated in recent recipes)
    try:
        ingredients = (soup.find("span", class_="wprm-recipe-key_ingredients wprm-block-text-normal")).getText()
    except:
        ingredients = ""
    # recipe name
    try:
        name = (soup.find("h2", class_="wprm-recipe-name wprm-block-text-bold")).getText()
    except:
        print(recipeUrl)
        name = ""
    # all ingredients list 
    allIngredients = soup.find_all("span", class_="wprm-recipe-ingredient-name")
    ingredientsList = " ".join([item.getText() for item in allIngredients])
    return [recipeUrl, name, course, cuisine, ingredients, ingredientsList]


# this function grabs the file of recipes if it exists, or scrapes the webpage 
# if the file does not exist, or if it has not been updated in the last day
# the function returns the dataframe from the file 
def getFile(pageNumber):
    
    # if recipe file exists and has been created in the last week, do not run again
    if path.exists("/Users/bridgetmoynihan/HBHRecipes.csv"):
        modTimesinceEpoc = os.path.getmtime("/Users/bridgetmoynihan/HBHRecipes.csv")
        modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
        date_time = parser.parse(modificationTime)

        # checking if file has been updated in the last week
        if (date.today() - timedelta(days=1) < date_time.date()): 
            rec = pd.read_csv("/Users/bridgetmoynihan/HBHRecipes.csv")

        # update file with new recipe
        # assume that file has been run before, so only grab recipes from 5 most recent pages
        # duplicates are handled later, so we are not concerned about this
        else:
            print("Need to add new recipes to file, please wait!")
            RECIPES = getRecipes(1)
            RECIPES['Course'] = RECIPES['Course'].apply(lambda x: x.split(sep=","))
            RECIPES = RECIPES.explode('Course')
            rec = pd.read_csv("/Users/bridgetmoynihan/HBHRecipes.csv")
            rec = pd.concat([RECIPES, rec], ignore_index=True)
            rec.to_csv("/Users/bridgetmoynihan/HBHRecipes.csv")
            print("Recipes have been added.")
            
    # if file does not exist, scrape entire webpage to grab all recipes    
    else:
        print("Recipe files have not been searched, please wait while they are scraped!")
        RECIPES = getRecipes(pageNumber)
        
            # splitting up recipes that have multiple courses, so its easier to filter
        RECIPES['Course'] = RECIPES['Course'].apply(lambda x: x.split(sep=","))
        RECIPES = RECIPES.explode('Course')
        rec = RECIPES
            # adding recipes dataframe to csv file
        RECIPES.to_csv("/Users/bridgetmoynihan/HBHRecipes.csv")
        print("Thanks for waiting! All recipes have been scraped :)")
    return rec

#### Text processing

#Stem and make lower case
def stemSentence(sentence):
    porter = PorterStemmer()
    token_words = word_tokenize(sentence)
    stem_sentence = [porter.stem(word) for word in token_words]
    return ' '.join(stem_sentence)    


# function for clustering the data
# it returns the cluster labels
def kMeansClustering(df, recipeName): 
    #Load data set
    ingredients = df['All Ingredients']
    
    #Remove stopwords, punctuation and numbers
    ingredientsClean = [remove_stopwords(x)\
            .translate(str.maketrans('','',string.punctuation))\
            .translate(str.maketrans('','',string.digits))\
            for x in ingredients]
    
    ingredientsCleanest = pd.Series([stemSentence(x) for x in ingredientsClean])
    
    # tokenize words in the ingredient list
    vectorizer_ntf = TfidfVectorizer(analyzer='word',ngram_range=(1,2))
    X_ntf = vectorizer_ntf.fit_transform(ingredientsCleanest)

    # clustering
    clusters = len(ingredientsCleanest)//5
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X_ntf)
    resultSet = pd.concat([ingredientsCleanest,pd.DataFrame(X_ntf.toarray(),columns=vectorizer_ntf.get_feature_names())],axis=1)
    resultSet['cluster'] = kmeans.predict(X_ntf)
    return resultSet
    

def main():
    # HBH url
    url = "https://www.halfbakedharvest.com/category/recipes/"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    ## find last page number, so we know how many pages to crawl
    navigation = soup.find_all("nav", class_="archive-pagination pagination")
    lists = navigation[0].find_all("li")
    string_pageNumber = lists[-2].getText()
    pageNumber = [int(s) for s in string_pageNumber.split() if s.isdigit()][0]

    print("HALFBAKED HARVEST RECIPE SUGGESTION!\n")
    
    rec = getFile(pageNumber) #dataframe of recipes
    
    # ask user for recipe link, and keep asking if its not a valid recipe
    inputGood = False
    while not inputGood:
        userRecipe = input("Please provide the link for the recipe from HBH you like:\n")
        try:
            recipeDetails = getRecipeDetails(userRecipe)
        except:
            print("Not a valid HBH recipe link!")
        else:
            inputGood = True
    print("~~~~~~~~~~~~~~~~~~~~~~~")
    
    recipeName = recipeDetails[1]
    course = recipeDetails[2].split(sep=",")
    # recipe does not have a specific course, so we look through ALL recipes with similar ingredients
    if course[0] == "":
        recNameIngredients = rec[["Link", "Recipe", "All Ingredients"]].drop_duplicates()
  
    # recipe does have specific course
    # get recipes from file that fall under that course 
    else:
        recipeSameCourse = rec[rec['Course'].isin(course)]
        recNameIngredients = recipeSameCourse[["Link", "Recipe", "All Ingredients"]].drop_duplicates()
   
    results = kMeansClustering(recNameIngredients, recipeName)
    recNameIngredients['cluster'] = list(results['cluster'])
    
    # cluster number of given recipe
    clusterNumber = int(recNameIngredients[recNameIngredients['Recipe'] == recipeName]["cluster"])
    
    # recipes in that same cluster
    similarRecipesLinks = list(recNameIngredients[recNameIngredients['cluster'] == clusterNumber]['Link'])
    similarRecipesNames = list(recNameIngredients[recNameIngredients['cluster'] == clusterNumber]['Recipe'])
    print("Based on the recipe you provided, here are some additional recipes we think you may like:\n ")
    
    # provide users with recipes that are similar 
    for i in range(len(similarRecipesLinks)):
        if similarRecipesNames[i] != recipeName:
            print(similarRecipesNames[i])
            print("Link: ", similarRecipesLinks[i])
            print()
            
main()
 