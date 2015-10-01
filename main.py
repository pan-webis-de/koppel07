# Robert Paßmann

NUMBER_ELIMINATE_FEATURES = 5 # each side, i.e. 3 max and 3 min
NUMBER_ITERATIONS = 10

import logging
import argparse
import jsonhandler
import unmasking 

from sklearn import svm
import numpy

jsonhandler.OUT_FNAME = "koppel07_" + str(NUMBER_ELIMINATE_FEATURES) + "e_" + \
                        str(NUMBER_ITERATIONS) + "i.json"

def tira(corpusdir, outputdir):
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()
    
    database = unmasking.Database()
    
    for candidate in jsonhandler.candidates:
        database.add_author(candidate)
        for training in jsonhandler.trainings[candidate]:
            logging.info("Reading training text '%s' of '%s'", training, candidate)
            text = unmasking.Text(jsonhandler.getTrainingText(candidate,training), 
                                  candidate + " " + training)
            database.add_text(candidate,text)
            text.create_chunks()
            
    database.calc_initial_feature_set()
    
    candidates = [] # this list shall contain the most likely candidates
            
    # runtime could surely be optimized
    for unknown in jsonhandler.unknowns:
        results = {} # dictionary containing the maximum difference (first and
                     # last iteration) for every author
        
        unknown_text = unmasking.Text(jsonhandler.getUnknownText(unknown),unknown)
        unknown_text.create_chunks()
        
        for candidate in jsonhandler.candidates:
            results[candidate] = 0
            
            for known_text in database.texts[candidate]:
                # reset the feature list, i.e. create a copy
                features = list(database.initial_feature_set)
                
                unmasking.select_chunks(unknown_text, known_text)
                
                # create label vector
                label = [0 for i in range(0,len(unknown_text.selected_chunks))] + \
                        [1 for i in range(0,len(known_text.selected_chunks))]
                label = numpy.array(label)
                label.reshape(len(unknown_text.selected_chunks) + len(known_text.selected_chunks), 1)
                
                
                #loop
                global NUMBER_ITERATIONS
                global NUMBER_ELIMINATE_FEATURES
                scores = []
                for i in range(0,NUMBER_ITERATIONS):
                    logging.info("Iteration #%s for texts '%s' and '%s'", 
                                 str(i+1), unknown, known_text.name)  
                    # create matrix
                    matrix = [ [ chunk.count(word)/unmasking.CHUNK_LENGTH 
                                 for word in features ] 
                               for chunk 
                               in (unknown_text.selected_chunks + known_text.selected_chunks)]
                    matrix = numpy.array(matrix)
                           
                    # svm 
                    classifier = svm.LinearSVC()
                    classifier.fit(matrix, label)
                    scores.append(classifier.score(matrix, label))
        
                    # delete strongest weighted features (NUMBER_ELIMINATE_FEATURES)
                    flist = classifier.coef_[0] #list of feature weights
        
                    # indices of maximum 3 values and minimum 3 values
                    delete = list(numpy.argsort(flist)[-NUMBER_ELIMINATE_FEATURES:]) \
                             + list(numpy.argsort(flist)[:NUMBER_ELIMINATE_FEATURES])
            
                    delete_features = []     
                    for i in delete:
                        delete_features.append(features[i])
            
                    logging.info("Delete %s", str(delete_features))
            
                    for feature in delete_features:
                        features.remove(feature)
                
                score = abs(scores[0] - scores[NUMBER_ITERATIONS-1])
                logging.info("Calculated a score of %s", str(score))
                if score > results[candidate]:
                    results[candidate] = score
                
        # which author has the biggest score?
        most_likely_author = max(results, key=results.get)
        logging.info("Most likely author is '%s' with a score of %s", 
                     most_likely_author, results[most_likely_author])
        candidates.append(most_likely_author)
        
    jsonhandler.storeJson(jsonhandler.unknowns, candidates, outputdir)

def main():
    parser = argparse.ArgumentParser(description='Tira submission for Delta.')
    parser.add_argument('-i', 
                        action='store',
                        help='Path to input directory')
    parser.add_argument('-o', 
                        action='store',
                        help='Path to output directory')
    
    args = vars(parser.parse_args())
    
    corpusdir = args['i']
    outputdir = args['o']
    
    tira(corpusdir, outputdir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s: %(message)s')
    main()