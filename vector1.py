#!/usr/bin/env python
import sys
import re
import os
import subprocess
import math
import numpy as np
from collections import defaultdict
from nltk.stem import PorterStemmer

"""
This program is almost directly conversion from vector1.prl.
The purpose is to help people save some time if they are more
comfortable of using python. However, don't trust this script,
and use it with caution. If you find any bug or possible mistake,
please email zzhang84@jhu.edu

Hint: search "To be implemented"

##########################################################
##  VECTOR1
##
##  Usage:   vector1     (no command line arguments)
##
##  The function &main_loop below gives the menu for the system.
##
##  This is an example program that shows how the core
##  of a vector-based IR engine may be implemented in Perl.
##
##  Some of the functions below are unimplemented, and some
##  are only partially implemented. Suggestions for additions
##  are given below and in the assignment handout.
##
##  You should feel free to modify this program directly,
##  and probably use this as a base for your implemented
##  extensions.  As with all assignments, the range of
##  possible enhancements is open ended and creativity
##  is strongly encouraged.
##########################################################


############################################################
## Program Defaults and Global Variables
############################################################
"""

DIR, file_name = os.path.split(os.path.realpath(__file__))
HOME = "."

token_docs = DIR + "/cacm"                 # tokenized cacm journals
corps_freq = DIR + "/cacm"                 # frequency of each token in the journ.
stoplist = DIR + "/common_words"           # common uninteresting words
titles = DIR + "/titles.short"             # titles of each article in cacm
token_qrys = DIR + "/query"                # tokenized canned querys
query_freq = DIR + "/query"                # frequency of each token in the querys
query_relv = DIR + "/query.rels"           # relevance of a journal entry to a given query

# these files are created in your "HOME" directory
token_intr = HOME + "/interactive"         # file created for interactive queries
inter_freq = HOME + "/interactive"         # frequency of each token in above

tmp_token = token_docs[:]
tmp_corps = corps_freq[:]
tmp_stoplist = stoplist[:]
tmp_titles = titles[:]
tmp_qrys = token_qrys[:]
tmp_freq = query_freq[:]
tmp_relv = query_relv[:]
tmp_intr = token_intr[:]
tmp_inter = inter_freq[:]

# doc_vector:
# An array of hashes, each array index indicating a particular
# query's weight "vector". (See more detailed below)

doc_vector = []

# qry_vector:
# An array of hashes, each array index indicating a particular query's
# weight "vector".

qry_vector = []

# docs_freq_hash
#
# dictionary which holds <token, frequency> pairs where
# docs_freq_hash[token] -> frequency
#   token     = a particular word or tag found in the cacm corpus
#   frequency = the total number of times the token appears in
#               the corpus.

docs_freq_hash = defaultdict(int)

# corp_freq_hash
#
# dictionary which holds <token, frequency> pairs where
# corp_freq_hash[token] -> frequency
#   token     = a particular word or tag found in the corpus
#   frequency = the total number of times the token appears per
#               document-- that is a token is counted only once
#               per document if it is present (even if it appears
#               several times within that document).

corp_freq_hash = defaultdict(int)

# stoplist_hash
#
# common list of uninteresting words which are likely irrelvant
# to any query.
# for given "word" you can do:   `if word in stoplist_hash` to check
# if word is in stop list
#
#   Note: this is an associative array to provide fast lookups
#         of these boring words

stoplist_hash = set()

# titles_vector
#
# vector of the cacm journal titles. Indexed in order of apperance
# within the corpus.

titles_vector  = []

# relevance_hash
#
# a hash of hashes where each <key, value> pair consists of
#
#   key   = a query number
#   value = a hash consisting of document number keys with associated
#           numeric values indicating the degree of relevance the
#           document has to the particular query.
#   relevance_hash[query_num][doc_num] = 1, if given query and doc is relavent

relevance_hash = defaultdict(lambda: defaultdict(int))





##########################################################################
## Variables will be cleared and re-initilized on certain function calls
##########################################################################

# doc_simula
#
# array used for storing query to document or document to document
# similarity calculations (determined by cosine_similarity, etc. )

doc_simula = []

# res_vector
#
# array used for storing the document numbers of the most relevant
# documents in a query to document or document to document calculation.

res_vector = []

list_of_synonyms = []
synonyms = {}

sys.stderr.write("INITIALIZING VECTORS ... \n")

thesures = False
if thesures:
    p = PorterStemmer()
    with open("synonyms_en.txt", 'r') as f:
        lines = f.read().decode('utf-8').split('\n')
        index = 0
        for line in lines:
            words_list = []
            words = line.split(',')
            for word in words:
                word = word.strip()
                if re.search('[a-zA-Z]', word):
                    word = word.lower()
                    word = p.stem(word)
                    words_list.append(word)
                    synonyms[word] = index
            list_of_synonyms.append(words_list)
            index+=1


##########################################################
##  INIT_FILES
##
##  This function specifies the names and locations of
##  input files used by the program.
##
##  Parameter:  $type   ("stemmed" or "unstemmed")
##
##  If $type == "stemmed", the filenames are initialized
##  to the versions stemmed with the Porter stemmer, while
##  in the default ("unstemmed") case initializes to files
##  containing raw, unstemmed tokens.
##########################################################


# You may change this to "stemmed" or "unstemmed"
def init_files(su="stemmed"):
    filename = su
    global token_docs
    global tmp_token
    token_docs = tmp_token
    global corps_freq
    global tmp_corps
    corps_freq = tmp_corps
    global stoplist
    global tmp_stoplist
    stoplist = tmp_stoplist
    global token_qrys
    global tmp_qrys
    token_qrys = tmp_qrys
    global query_freq
    global tmp_freq
    query_freq = tmp_freq
    global token_intr
    global tmp_intr
    token_intr = tmp_intr
    global inter_freq
    global tmp_inter
    inter_freq = tmp_inter

    if filename == "stemmed":
        token_docs += ".stemmed"
        corps_freq += ".stemmed.hist"
        stoplist   += ".stemmed"
        token_qrys += ".stemmed"
        query_freq += ".stemmed.hist"
        token_intr += ".stemmed"
        inter_freq += ".stemmed.hist"
    else:
        token_docs += ".tokenized"
        corps_freq += ".tokenized.hist"
        token_qrys += ".tokenized"
        query_freq += ".tokenized.hist"
        token_intr += ".tokenized"
        inter_freq += ".tokenized.hist"

##########################################################
##  INIT_CORP_FREQ
##
##  This function reads in corpus and document frequencies from
##  the provided histogram file for both the document set
##  and the query set. This information will be used in
##  term weighting.
##
##  It also initializes the arrays representing the stoplist,
##  title list and relevance of document given query.
##########################################################


def init_corp_freq():
    for line in open(corps_freq, 'r'):
        per_data = line.strip().split()
        if len(per_data) == 3:
            doc_freq, corp_freq, term = line.strip().split()
            docs_freq_hash[term] = int(doc_freq)
            corp_freq_hash[term] = int(corp_freq)

    for line in open(query_freq, 'r'):
        per_data = line.strip().split()
        if len(per_data) == 3:
            doc_freq, corp_freq, term = line.strip().split()
            docs_freq_hash[term] += int(doc_freq)
            corp_freq_hash[term] += int(corp_freq)

    for line in open(stoplist, 'r'):
        if line:
            stoplist_hash.add(line.strip())

    # push one empty value onto titles_vector
    # so that indices correspond with title numbers.
    # title looks like:
    # titles_vector[3195] = "Gorn    #  Reiteration of ACM Policy Toward Standardization"
    titles_vector.append("")

    for line in open(titles, 'r'):
        if line:
            titles_vector.append(line.strip())

    for line in open(query_relv, 'r'):
        if line:
            qry_num, rel_doc = map(int, line.strip().split())
            relevance_hash[qry_num][rel_doc] = 1


##########################################################
##  INIT_DOC_VECTORS
##
##  This function reads in tokens from the document file.
##  When a .I token is encountered, indicating a document
##  break, a new vector is begun. When individual terms
##  are encountered, they are added to a running sum of
##  term frequencies. To save time and space, it is possible
##  to normalize these term frequencies by inverse document
##  frequency (or whatever other weighting strategy is
##  being used) while the terms are being summed or in
##  a posthoc pass.  The 2D vector array
##
##    doc_vector[ doc_num ][ term ]
##
##  stores these normalized term weights.
##
##  It is possible to weight different regions of the document
##  differently depending on likely importance to the classification.
##  The relative base weighting factors can be set when
##  different segment boundaries are encountered.
##
##  This function is currently set up for simple TF weighting.
##########################################################


def init_doc_vectors(title=3, key=4, abstr=3, authr=1, tf="idf", all_tokens=False):
    # Make sure doc_vector is emtpy
    global doc_vector
    doc_vector = []

    doc_num = 0

    TITLE_BASE_WEIGHT = title     # weight given a title token
    KEYWD_BASE_WEIGHT = key     # weight given a key word token
    ABSTR_BASE_WEIGHT = abstr     # weight given an abstract word token
    AUTHR_BASE_WEIGHT = authr     # weight given an an author token

    BASE_WEIGHT = {".T" : TITLE_BASE_WEIGHT, ".K" : KEYWD_BASE_WEIGHT,\
                    ".W" : ABSTR_BASE_WEIGHT, ".A" : AUTHR_BASE_WEIGHT}

    tweight = 0
    # push one empty value onto qry_vectors so that
    # indices correspond with query numbers
    doc_vector.append(defaultdict(int))

    for word in open(token_docs, 'r'):
        word = word.strip()
        if not word or word == ".I 0":
            continue  # Skip empty line
        if word[:2] == ".I":
            new_doc_vec = defaultdict(int)
            doc_vector.append(new_doc_vec)
            doc_num += 1
        elif word in BASE_WEIGHT:
            tweight = BASE_WEIGHT[word]
        elif re.search("[a-zA-Z]", word):
            if all_tokens:
                if docs_freq_hash[word] == 0:
                    exit("ERROR: Document frequency of zero: " + word + \
                         " (check if token file matches corpus_freq file\n")
                new_doc_vec[word] += tweight
            elif word not in stoplist_hash:
                if docs_freq_hash[word] == 0:
                    exit("ERROR: Document frequency of zero: " + word + \
                         " (check if token file matches corpus_freq file\n")
                if tf == "Boolean":
                    new_doc_vec[word] = 1
                else:
                    new_doc_vec[word] += tweight

    global total_docs
    total_docs = doc_num

    if tf == "idf":
        for docn in range(1, len(doc_vector)):
            for term in doc_vector[docn]:
                doc_vector[docn][term] *= math.log(float(total_docs) / docs_freq_hash[term])


###############################################################
##  INIT_QRY_VECTORS
##
##  This function should be nearly identical to the step
##  for initializing document vectors.
##
##  This function is currently set up for simple TF weighting.
###############################################################


def init_qry_vectors(tf="raw"):
    # Make sure qry_vector is empty
    global qry_vector
    qry_vector = []

    qry_num = 0

    QUERY_BASE_WEIGHT = 2
    QUERY_AUTH_WEIGHT = 2

    QUERY_WEIGHT = {".W" : QUERY_BASE_WEIGHT, ".A" : QUERY_AUTH_WEIGHT}

    # push one empty value onto qry_vectors so that
    # indices correspond with query numbers
    qry_vector.append(defaultdict(int))

    for word in open(token_qrys, 'r'):
        word = word.strip()
        if not word or word == ".I 0":
            continue  # Skip empty line

        if word[:2] == ".I":
            new_doc_vec = defaultdict(int)
            qry_vector.append(new_doc_vec)
            qry_num += 1
        elif word in QUERY_WEIGHT:
            tweight = QUERY_WEIGHT[word]
        elif word not in stoplist_hash and re.search('[a-zA-Z]', word):
            if docs_freq_hash[word] == 0:
                sys.stderr.write("ERROR: Document frequency of zero: " + word + \
                              " (check if token file matches corpus_freq file\n")
                exit(1)
            new_doc_vec[word] += tweight

    global total_qrys
    total_qrys = qry_num

    if tf == "idf2":
        for qryn in range(1, len(qry_vector)):
            for term in qry_vector[qryn]:
                qry_vector[qryn][term] *= math.log(float(total_qrys) / docs_freq_hash[term])


##########################################################
## MAIN_LOOP
##
## Parameters: currently no explicit parameters.
##             performance dictated by user imput.
##
## Initializes document and query vectors using the
## input files specified in &init_files. Then offers
## a menu and switch to appropriate functions in an
## endless loop.
##
## Possible extensions at this level:  prompt the user
## to specify additional system parameters, such as the
## similarity function to be used.
##
## Currently, the key parameters to the system (stemmed/unstemmed,
## stoplist/no-stoplist, term weighting functions, vector
## similarity functions) are hardwired in.
##
## Initializing the document vectors is clearly the
## most time consuming section of the program, as 213334
## to 258429 tokens must be processed, weighted and added
## to dynamically growing vectors.
##
##########################################################



# if you have trouble of running this (perphas you copy hw2 to run on your
# local machine). Try to 1) change the path of file directory 2) change shebang
# (e.g. you may use "#!/usr/bin/perl" instead of "#!/usr/local/bin/perl")
# 3) remake some c files. (make clean; make   in "hw2" and "stemmer" directory)



def main():
    init_files()
    init_corp_freq()
    init_doc_vectors()
    init_qry_vectors()

    menu = \
    "============================================================\n"\
    "==      Welcome to the 600.466 Vector-based IR Engine       \n"\
    "==                                                          \n"\
    "==      Total Documents: {0}                                \n"\
    "==      Total Queries: {1}                                  \n"\
    "============================================================\n"\
    "                                                            \n"\
    "OPTIONS:                                                    \n"\
    "  1 = Find documents most similar to a given query or document\n"\
    "  2 = Compute precision/recall for the full query set         \n"\
    "  3 = Compute cosine similarity between two queries/documents \n"\
    "  4 = Quit                                                    \n"\
    "                                                              \n"\
    "============================================================\n".format(total_docs, total_qrys)

    while True:
        sys.stderr.write(menu)
        option = raw_input("Enter Option: ")
        if option == "1":
            get_and_show_retrieved_set()
        elif option == "2":
            full_precision_recall_test()
        elif option == "3":
            do_full_cosine_similarity()
        elif option == "4":
            exit(0)
        else:
            sys.stderr.write("Input seems not right, try again\n")


##########################################################
## GET_AND_SHOW_RETRIEVED_SET
##
##  This function requests key retrieval parameters,
##  including:
##
##  A) Is a query vector or document vector being used
##     as the retrieval seed? Both are vector representations
##     but they are stored in different data structures,
##     and one may optionally want to treat them slightly
##     differently.
##
##  B) Enter the number of the query or document vector to
##     be used as the retrieval seed.
##
##     Alternately, one may wish to request a new query
##     from standard input here (and call the appropriate
##     tokenization, stemming and term-weighting routines).
##
##  C) Request the maximum number of retrieved documents
##     to display.
##
##########################################################

def get_and_show_retrieved_set():
    menu = "Find documents similar to:  \n"\
           "(1) a query from 'query.raw'\n"\
           "(2) an interactive query    \n"\
           "(3) another document        \n"\
           "(4) query from keyboard     \n"
    sys.stderr.write(menu)
    comp_type = raw_input("Choice: ")

    if re.match("[123]\s$", comp_type):
        sys.stderr.write("The input is not valid. By default use option 1\n")
        comp_type = "1"

    if comp_type != "2" and comp_type != "4":
        vect_num = int(raw_input("Target Document/Query number: "))

    max_show = int(raw_input("Show how many matching documents (e.g. 20): "))

    if comp_type == "3":
        sys.stderr.write("Document to Document comparison\n")
        get_retrieved_set(doc_vector[vect_num])
        shw_retrieved_set(max_show, vect_num, doc_vector[vect_num], "Document")
    elif comp_type == "2":
        sys.stderr.write("Interactive Query to Document comparison\n")
        int_vector = set_interact_vec()
        get_retrieved_set(int_vector)
        shw_retrieved_set(max_show, 0, int_vector, "Interactive Query")
    elif comp_type == "4":
        sys.stderr.write("Keyboard input Query to Document Comparison\n")
        int_vector = keyboard_query()
        print int_vector
        get_retrieved_set(int_vector)
        shw_retrieved_set(max_show, 0, int_vector, "Interactive Query")
    else:
        sys.stderr.write("Query to Document comparison\n")
        get_retrieved_set(qry_vector[vect_num])
        shw_retrieved_set(max_show, vect_num, qry_vector[vect_num], "Query")
        comp_recall(relevance_hash[vect_num], vect_num)
        show_relvnt(relevance_hash[vect_num], vect_num, qry_vector[vect_num])


def get_retrieved_set(my_doc_vector):
    pass

def shw_retrieved_set(max_show, vect_num_, my_doc_vector, query_type):
    pass

def set_interact_vec():
    subprocess.call(DIR + "/interactive.prl")
    int_vector = []
    qry_num = 0

    QUERY_BASE_WEIGHT = 2
    QUERY_AUTH_WEIGHT = 2

    QUERY_WEIGHT = {".W" : QUERY_BASE_WEIGHT, ".A" : QUERY_AUTH_WEIGHT}


    for word in open(token_intr, 'r'):
        word = word.strip()
        if not word or word == ".I 0":
            continue  # Skip empty line

        if word[:2] == ".I":
            new_doc_vec = defaultdict(int)
            int_vector.append(new_doc_vec)
            qry_num += 1
        elif word in QUERY_WEIGHT:
            tweight = QUERY_WEIGHT[word]
        elif word not in stoplist_hash and re.search('[a-zA-Z]', word):
            if docs_freq_hash[word] == 0:
                sys.stderr.write("ERROR: Document frequency of zero: " + word + \
                                " (check if token file matches corpus_freq file\n")
                exit(1)
            new_doc_vec[word] += tweight

    return int_vector


def keyboard_query():
    p = PorterStemmer()
    qry = raw_input("Type Your Query: ")
    list_words = qry.strip().split()
    QUERY_WEIGHT = 2
    usr_qry_vect = defaultdict(int)
    for i in range(len(list_words)):
        word = p.stem(list_words[i])
        if word not in stoplist_hash:
            if word not in usr_qry_vect.keys():
                usr_qry_vect[word] = QUERY_WEIGHT
            else:
                usr_qry_vect[word] += QUERY_WEIGHT

    if thesures:
        new_vect = defaultdict(int)
        for key in usr_qry_vect:
            new_vect[key] = usr_qry_vect[key]
            if key in synonyms:
                similar_words_list = list_of_synonyms[synonyms[key]]
                for s_word in similar_words_list:
                    if s_word not in stoplist_hash and re.search("[a-zA-z]", s_word):
                        if corp_freq_hash[s_word] > 1:
                            new_vect[s_word] = usr_qry_vect[key]
        return new_vect

    return usr_qry_vect


###########################################################
## GET_RETRIEVED_SET
##
##  Parameters:
##
##  my_qry_vector    - the query vector to be compared with the
##                  document set. May also be another document
##                  vector.
##
##  This function computes the document similarity between the
##  given vector "my_qry_vector" and all vectors in the document
##  collection storing these values in the array "doc_simula"
##
##  An array of the document numbers is then sorted by this
##  similarity function, forming the rank order of documents
##  for use in the retrieval set.
##
##  The similarity will be
##  sorted in descending order.
##########################################################



def get_retrieved_set(my_qry_vector, similarity="cosine"):
    # "global" variable might not be a good choice in python, but this
    # makes us consistant with original perl script
    global doc_simula, res_vector

    tot_number = len(doc_vector) - 1

    doc_simula = []   # insure that storage vectors are empty before we
    res_vector = []   # calculate vector similarities

    doc_simula.append(0)

    if similarity == "cosine":
        for index in range(1, tot_number + 1):
            doc_simula.append(cosine_sim_a(my_qry_vector, doc_vector[index]))
    elif similarity == "dice":
        for index in range(1, tot_number + 1):
            doc_simula.append(dice_sim(my_qry_vector, doc_vector[index]))

    res_vector = sorted(range(1, tot_number + 1), key = lambda x: -doc_simula[x])


############################################################
## SHW_RETRIEVED_SET
##
## Assumes the following global data structures have been
## initialized, based on the results of "get_retrieved_set".
##
## 1) res_vector - contains the document numbers sorted in
##                  rank order
## 2) doc_simula - The similarity measure for each document,
##                  computed by &get_retrieved_set.
##
## Also assumes that the following have been initialized in
## advance:
##
##       titles[ doc_num ]    - the document title for a
##                                document number, $doc_num
##       relevance_hash[ qry_num ][ doc_num ]
##                              - is doc_num relevant given
##                                query number, qry_num
##
## Parameters:
##   max_show   - the maximum number of matched documents
##                 to display.
##   qry_num    - the vector number of the query
##   qry_vect   - the query vector (passed by reference)
##   comparison - "Query" or "Document" (type of vector
##                 being compared to)
##
## In the case of "Query"-based retrieval, the relevance
## judgements for the returned set are displayed. This is
## ignored when doing document-to-document comparisons, as
## there are nor relevance judgements.
##
############################################################


def shw_retrieved_set(max_show, qry_num, my_qry_vector, comparison):
    menu = "   ************************************************************\n"\
           "        Documents Most Similar To {0} number {1}       \n"\
           "   ************************************************************\n"\
           "   Similarity   Doc#  Author      Title                        \n"\
           "   ==========   ===  ========     =============================\n".format(comparison, qry_num)
    sys.stderr.write(menu)
    for index in range(max_show + 1):
        ind = res_vector[index]
        if comparison == "Query" and relevance_hash[qry_num][ind]:
            sys.stderr.write("* ")
        else:
            sys.stderr.write("  ")

        similarity = doc_simula[ind]
        title = titles_vector[ind][:47]

        sys.stderr.write(" {0:10.8f}   {1}\n".format(similarity, title))

    show_terms = raw_input("\nShow the terms that overlap between the query and"\
        "retrieved docs (y/n): ").strip()
    if show_terms != 'n' and show_terms != 'N':
        for index in range(max_show + 1):
            ind = res_vector[index]

            show_overlap(my_qry_vector, doc_vector[ind], qry_num, ind)

            if (index % 5 == 4):
                cont = raw_input("\nContinue (y/n)?: ").strip()
                if cont == 'n' or cont == 'N':
                    break


##########################################################
## COMPUTE_PREC_RECALL
##
## Like "shw_retrieved_set", this function makes use of the following
## data structures which may either be passed as parameters or
## used as global variables. These values are set by the function
## &get_retrieved_set.
##
## 1) doc_simila[ rank ] - contains the document numbers sorted
##                          in rank order based on the results of
##                          the similarity function
##
## 2) res_vector[ docn ] - The similarity measure for each document,
##                          relative to the query vector ( computed by
##                          "get_retrieved_set").
##
## Also assumes that the following have been initialzied in advance:
##       titles[ docn ]       - the document title for a document
##                                number $docn
##       relevance_hash[ qvn ][ docn ]
##                              - is $docn relevant given query number
##                                $qvn
##
##  The first step of this function should be to take the rank ordering
##  of the documents given a similarity measure to a query
##  (i.e. the list docs_sorted_by_similarity[rank]) and make a list
##  of the ranks of just the relevant documents. In an ideal world,
##  if there are k=8 relevant documents for a query, for example, the list
##  of rank orders should be (1 2 3 4 5 6 7 8) - i.e. the relevant documents
##  are the top 8 entries of all documents sorted by similarity.
##  However, in real life the relevant documents may be ordered
##  much lower in the similarity list, with rank orders of
##  the 8 relevant of, for example, (3 27 51 133 159 220 290 1821).
##
##  Given this list, compute the k (e.g. 8) recall/precison pairs for
##  the list (as discussed in class). Then to determine precision
##  at fixed levels of recall, either identify the closest recall
##  level represented in the list and use that precision, or
##  do linear interpolation between the closest values.
##
##  This function should also either return the various measures
##  of precision/recall specified in the assignment, or store
##  these values in a cumulative sum for later averaging.
##########################################################


def comp_recall(relevance_qry_hash, vect_num):
    # List of ranks of relevant documents
    global docs_sorted_by_similarity
    docs_sorted_by_similarity = []

    for index, docn in enumerate(res_vector, 1):
        if relevance_qry_hash[docn]:
            docs_sorted_by_similarity.append(index)

    # Total number of relevant documents
    Rel = len(docs_sorted_by_similarity)
    # Total number of documents
    N = total_docs

    # The k recall/precision pairs for the list
    global recall
    global precision
    recall = []
    precision = []

    for index in range(Rel):
        recall.append(float(index + 1) / Rel)
        precision.append(float(index + 1) / docs_sorted_by_similarity[index])

    # Linear interpolation between the closest values
    prec_25 = np.interp(0.25, recall, precision)
    prec_50 = np.interp(0.50, recall, precision)
    prec_75 = np.interp(0.75, recall, precision)
    prec_100 = np.interp(1.00, recall, precision)

    prec_mean1 = (prec_25 + prec_50 + prec_75) / 3.0
    prec_sum = 0
    for i in range(10):
        prec_sum += np.interp((i + 1) / 10.0, recall, precision)
    prec_mean2 = prec_sum / 10.0

    rank_sum = 0.0
    r_index_sum = 0.0
    rank_logsum = 0.0
    p_index_sum = 0.0
    for i in range(Rel):
        rank_sum += docs_sorted_by_similarity[i]
        r_index_sum += i + 1
        rank_logsum += math.log(docs_sorted_by_similarity[i])
        p_index_sum += math.log(i + 1)

    recall_norm = 1.0 - ((rank_sum - r_index_sum) / (Rel * (N - Rel)))
    prec_norm = 1.0 - ((rank_logsum - p_index_sum)/(N*math.log(N) - (N-Rel)*math.log((N-Rel)) - Rel*math.log(Rel)))

    return prec_25, prec_50, prec_75, prec_100, prec_mean1, prec_mean2, recall_norm, prec_norm

##########################################################
## SHOW_RELVNT
##
## UNIMPLEMENTED
##
## This function should take the rank orders and similarity
## arrays described in &show_retrieved_set and &comp_recall
## and print out only the relevant documents, in an order
## and manner of presentation very similar to &show_retrieved_set.
##########################################################


def show_relvnt(relevance_qry_hash, vect_num, my_qry_vector):
    menu = "   **************************************************************\n"\
           "                      Documents Relevant To {0}                  \n"\
           "   **************************************************************\n"\
           "    i    Rank    Doc#  Author      Title                        \n"\
           "   ===   =====   ===  ========     =============================\n".format(vect_num)
    sys.stderr.write(menu)

    index = 0
    for docn in res_vector:
        if relevance_qry_hash[docn]:
            sys.stderr.write("* ")
            rank = res_vector.index(docn) + 1
            title = titles_vector[docn][:47]
            index += 1
            sys.stderr.write(" {0:2}    {1:4}    {2}\n".format(index, rank, title))


########################################################
## SHOW_OVERLAP
##
## Parameters:
##  - Two vectors (qry_vect and doc_vect), passed by
##    reference.
##  - The number of the vectors for display purposes
##
## PARTIALLY IMPLEMENTED:
##
## This function should show the terms that two vectors
## have in common, the relative weights of these terms
## in the two vectors, and any additional useful information
## such as the document frequency of the terms, etc.
##
## Useful for understanding the reason why documents
## are judged as relevant.
##
## Present in a sorted order most informative to the user.
##
########################################################


def show_overlap(my_qry_vector, my_doc_vector, qry_num, doc_num):
    info = "============================================================\n"\
           "{0:15s}  {1:8d}   {2:8d}\t{3}\n"\
           "============================================================\n".format(
            "Vector Overlap", qry_num, doc_num, "Docfreq")
    sys.stderr.write(info)
    for term_one, weight_one in my_qry_vector.items():
        if my_doc_vector.get(term_one, 0):
            info =  "{0:15s}  {1:8d}   {2:8f}\t{3}\n".format(
                term_one, weight_one, my_doc_vector[term_one], docs_freq_hash[term_one])
            sys.stderr.write(info)


########################################################
## DO_FULL_COSINE_SIMILARITY
##
##  Prompts for a document number and query number,
##  and then calls a function to show similarity.
##
##  Could/should be expanded to handle a variety of
##  similarity measures.
########################################################


def do_full_cosine_similarity():
    num_one = int(raw_input("\n1st is Query number: ").strip())
    num_two = int(raw_input("\n2nd is Document number: ").strip())

    full_cosine_similarity(qry_vector[num_one], doc_vector[num_two], num_one, num_two)


########################################################
## FULL_COSINE_SIMILARITY
##
## This function should compute cosine similarity between
## two vectors and display the information that went into
## this calculation, useful for debugging purposes.
## Similar in structure to &show_overlap.
########################################################

def full_cosine_similarity(my_qry_vector, my_doc_vector, num_one, num_two):
    info = "============================================================\n"\
           "  Full Cosine Similarity Between Query {0} And Document {1} \n"\
           "============================================================\n".format(num_one, num_two)
    sys.stderr.write(info)

    vec1 = my_qry_vector
    vec2 = my_doc_vector

    vec1_norm = sum(v * v for v in vec1.values())
    vec2_norm = sum(v * v for v in vec2.values())

    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    # calculate the cross product
    cross_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1.keys())
    cosine_sim = cross_product / math.sqrt(vec1_norm * vec2_norm)

    print "Vector 1 Normalized: %.2f" % vec1_norm
    print "Vector 2 Normalized: %.2f" % vec2_norm
    print "Cross Product of Vectors: %.2f" % cross_product
    print "\n Cosine Similarity: %f" % cosine_sim
    return cosine_sim


##########################################################
##  FULL_PRECISION_RECALL_TEST
##
##  This function should test the various precision/recall
##  measures discussed in the assignment and store cumulative
##  statistics over all queries.
##
##  As each query takes a few seconds to process, print
##  some sort of feedback for each query so the user
##  has something to watch.
##
##  It is helpful to also log this information to a file.
##########################################################

def full_precision_recall_test():

    menu = "   *************************************************************************************************\n"\
           "                                       Full Precision Recall Test                                   \n"\
           "   *************************************************************************************************\n"\
           "   Permutation Name              P_.25   P_.50   P_.75   P_1.0   P_mean1   P_mean2   P_norm   R_norm\n"\
           "   ===========================   =====   =====   =====   =====   =======   =======   ======   ======\n"
    sys.stderr.write(menu)

    for option in range(12):
        if option == 0:
            name = "Raw TF Weighting"
            init_doc_vectors(3, 4, 3, 1, "raw")
            option_print(name, "cosine")
        elif option == 1:
            name = "TF IDF Weighting"
            init_doc_vectors(3, 4, 3, 1, "idf")
            option_print(name, "cosine")
        elif option == 2:
            name = "Boolean Weighting"
            init_doc_vectors(3, 4, 3, 1, "Boolean")
            option_print(name, "cosine")
        elif option == 3:
            name = "Cosine Similarity"
            init_doc_vectors()
            option_print(name, "cosine")
        elif option == 4:
            name = "Dice Similarity"
            init_doc_vectors()
            option_print(name, "dice")
        elif option == 5:
            name = "Unstemmed Tokens"
            init_files("unstemmed")
            init_corp_freq()
            init_doc_vectors()
            init_qry_vectors()
            option_print(name, "cosine")
        elif option == 6:
            name = "Porter Stemmer"
            init_files("stemmed")
            init_corp_freq()
            init_doc_vectors()
            init_qry_vectors()
            option_print(name, "cosine")
        elif option == 7:
            name = "Exclude Stopwards"
            init_doc_vectors(3, 4, 3, 1, "idf", False)
            option_print(name, "cosine")
        elif option == 8:
            name = "Include All Tokens"
            init_doc_vectors(3, 4, 3, 1, "idf", True)
            option_print(name, "cosine")
        elif option == 9:
            name = "Weight Equally"
            init_doc_vectors(1, 1, 1, 1)
            option_print(name, "cosine")
        elif option == 10:
            name = "Relative Weights 3,4,3,1"
            init_doc_vectors(3, 4, 3, 1)
            option_print(name, "cosine")
        elif option == 11:
            name = "Relative Weights 1,1,1,4"
            init_doc_vectors(1, 1, 1, 4)
            option_print(name, "cosine")


def option_print(name, similarity):
    P25 = []
    P50 = []
    P75 = []
    P1 = []
    Pmean1 = []
    Pmean2 = []
    Pnorm = []
    Rnorm = []

    for i in range(1, len(qry_vector)):
        get_retrieved_set(qry_vector[i], similarity)
        p25, p50, p75, p1, pmean1, pmean2, rnorm, pnorm = comp_recall(relevance_hash[i], i)
        P25.append(p25)
        P50.append(p50)
        P75.append(p75)
        P1.append(p1)
        Pmean1.append(pmean1)
        Pmean2.append(pmean2)
        Pnorm.append(pnorm)
        Rnorm.append(rnorm)

    sys.stderr.write("   {:24s}      {:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.4f}    {:.4f}    {:.4f}   {:.4f} \n".format(
        name, np.mean(P25), np.mean(P50), np.mean(P75), np.mean(P1), np.mean(Pmean1), np.mean(Pmean2), np.mean(Pnorm),
        np.mean(Rnorm)))


########################################################
## COSINE_SIM_A
##
## Computes the cosine similarity for two vectors
## represented as associate arrays. You can also pass the
## norm as parameter
##
## Note: You may do it in a much efficient way like
## precomputing norms ahead or using packages like
## "numpy", below provide naive implementation of that
########################################################

def cosine_sim_a(vec1, vec2, vec1_norm = 0.0, vec2_norm = 0.0):
    if not vec1_norm:
        vec1_norm = sum(v * v for v in vec1.values())
    if not vec2_norm:
        vec2_norm = sum(v * v for v in vec2.values())

    # save some time of iterating over the shorter vec
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    # calculate the cross product
    cross_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1.keys())
    return cross_product / math.sqrt(vec1_norm * vec2_norm)


def dice_sim(vec1, vec2, vec1_norm = 0.0, vec2_norm = 0.0):
    if not vec1_norm:
        vec1_norm = sum(v * v for v in vec1.values())
    if not vec2_norm:
        vec2_norm = sum(v * v for v in vec2.values())

    # save some time of iterating over the shorter vec
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    # calculate the cross product
    cross_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1.keys())
    return 2 * (cross_product / (vec1_norm + vec2_norm))

########################################################
##  COSINE_SIM_B
##  Same thing, but to be consistant with original perl
##  script, we add this line
########################################################


def cosine_sim_b(vec1, vec2, vec1_norm = 0.0, vec2_norm = 0.0):
    return cosine_sim_a(vec1, vec2, vec1_norm, vec2_norm)


if __name__ == "__main__": main()






