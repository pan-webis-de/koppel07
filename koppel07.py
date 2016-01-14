# Robert Paßmann

NUMBER_ELIMINATE_FEATURES = 3 # number of features eliminated from each extreme
NUMBER_ITERATIONS = 10

# BEST CONFIGURATIONS:
# PAN12I, ELIMINATE = 3, ITERATIONS = 10

import logging
import argparse
import jsonhandler
import unmasking 
import sys

from sklearn import svm
import numpy

def tira(corpusdir, outputdir):
    # load training data
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()
    
    database = unmasking.Database()
    
    for candidate in jsonhandler.candidates:
        database.add_author(candidate)
        for training in jsonhandler.trainings[candidate]:
            logging.info("Reading training text '%s' of '%s'", training, candidate)
            text = unmasking.Text(jsonhandler.getTrainingText(candidate,training), 
                                  candidate + " " + training)
            try: 
                text.create_chunks()
                database.add_text(candidate,text)
            except:
                #logging.info("Text size too small. Skip this text.")
                logging.warning("Text too small. Exit.")
                sys.exit()
            
    database.calc_initial_feature_set()
    
    candidates = [] # this list shall contain the most likely candidates
            
    # We use the unmasking procedure to compare all unknown texts to all 
    # enumerated texts of known authorship and then decide which fit best.
    # runtime could surely be optimized
    for unknown in jsonhandler.unknowns:
        try:
            results = {} # dictionary containing the maximum difference (first and
                         # last iteration) for every author
        
            # load the unknown text and create the chunks which are used
            # for the unmasking process
            unknown_text = unmasking.Text(jsonhandler.getUnknownText(unknown),unknown)
            unknown_text.create_chunks()
        
            for candidate in jsonhandler.candidates:
                results[candidate] = float("inf")
            
                for known_text in database.texts[candidate]:
                    # reset the feature list, i.e. create a copy of the initial list
                    features = list(database.initial_feature_set)
                
                    # randomly select equally many chunks from each text
                    unmasking.select_chunks(unknown_text, known_text)
                
                    # create label vector 
                    # (0 -> chunks of unknown texts, 1 -> chunks of known texts)
                    label = [0 for i in range(0,len(unknown_text.selected_chunks))] + \
                            [1 for i in range(0,len(known_text.selected_chunks))]
                    label = numpy.array(label)
                    # the reshape is necessary for the classifier
                    label.reshape(len(unknown_text.selected_chunks) + len(known_text.selected_chunks), 1)
                
                    #loop
                    global NUMBER_ITERATIONS
                    global NUMBER_ELIMINATE_FEATURES
                    scores = []
                    for i in range(0,NUMBER_ITERATIONS):
                        logging.info("Iteration #%s for texts '%s' and '%s'", 
                                     str(i+1), unknown, known_text.name)  
                        # Create the matrix containing the relative word counts
                        # in each chunk (for the selected features)
                        matrix = [ [ chunk.count(word)/unmasking.CHUNK_LENGTH 
                                     for word in features ] 
                                   for chunk 
                                   in (unknown_text.selected_chunks + known_text.selected_chunks)]
                        matrix = numpy.array(matrix)
                           
                        # Get a LinearSVC classifier and its score (i.e. accuracy 
                        # in the training data). Save this score as a point in the
                        # scores curve. (We want to select the curve with the
                        # steepest decrease)
                        classifier = svm.LinearSVC()
                        classifier.fit(matrix, label)
                        scores.append(classifier.score(matrix, label))
        
                        # a list of all feature weights 
                        flist = classifier.coef_[0]
        
                        # Now, we have to delete the strongest weighted features
                        # (NUMBER_ELIMINATE_FEATURES) from each side.
                        # indices of maximum 3 values and minimum 3 values
                        delete = list(numpy.argsort(flist)[-NUMBER_ELIMINATE_FEATURES:]) \
                                 + list(numpy.argsort(flist)[:NUMBER_ELIMINATE_FEATURES])
            
                        # We cannot directly use the delete list to eliminate from 
                        # the features list since peu-a-peu elimination changes
                        # the indices.
                        delete_features = []     
                        for i in delete:
                            delete_features.append(features[i])
            
                        logging.info("Delete %s", str(delete_features))
            
                        for feature in delete_features:
                            # a single feature could appear twice in the delete list
                            if feature in features: 
                                features.remove(feature)
                
                    # The scores list is now the graph we use to get our results
                    # Therefore, compare with previous scores.
                    score = unmasking.curve_score(scores)
                    logging.info("Calculated a score of %s", str(score))
                    if score < results[candidate]:
                        results[candidate] = score
                
            # Which author has the biggest score?
            most_likely_author = min(results, key=results.get)
            logging.info("Most likely author is '%s' with a score of %s", 
                         most_likely_author, results[most_likely_author])
            candidates.append(most_likely_author)
        except:
            candidates.append("FILE_TO_SMALL")
        
    # save everything in the specified directory
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
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(levelname)s: %(message)s')
    main()
