from naive_bayes import BayesClassificator

if __name__=="__main__":
    x_train, x_test, y_train, y_test = BayesClassificator.get_data()
    
    classificator = BayesClassificator()
    classificator.fit(x_train,y_train)
    