# coding=utf-8
import numpy as np
import pandas as pd
import sys
import optparse
from lxml import etree
from sklearn import metrics

def main(argv=None):
    # parse the input
    parser = optparse.OptionParser()
    parser.add_option('-g')
    parser.add_option('-t')
    options, args = parser.parse_args()
    gold_file_name = options.g
    test_file_name = options.t

    # process file with gold markup
    gold={}

    tree = etree.parse(gold_file_name)
    doc = tree.getroot()
    itemlist = doc.findall("review")

    target_categories = []

    test_ids = []
    for itm in itemlist:
        review_id = itm.get("id")
        test_ids.append(int(review_id))
        categories = itm.find("categories").findall("category")
        for xml_categ in categories:
            category = xml_categ.get("name")
            target_categories.append(category)
            sentiment = xml_categ.get("sentiment")
            gold[review_id + "_" + category] = sentiment

    # process file with participant markup
    test = {}
    tree = etree.parse(test_file_name)
    doc = tree.getroot()
    itemlist = doc.findall("review")

    for itm in itemlist:
        review_id = int(itm.get("id"))

        if review_id in test_ids: #it's test review
            categories = itm.find("categories").findall("category")
            for xml_categ in categories:
                category = xml_categ.get("name")
                sentiment = xml_categ.get("sentiment")
                test[str(review_id) + "_" + category] = sentiment

    avgs = []
    out2write = ["","id\tactual\tpredicted"]

    for categ in list(set(target_categories)):
        print categ + "\t",
    print "avg_all"
    for categ in list(set(target_categories)):
        actual, predicted = [], []
        for key in gold:
            if categ!=key.split('_')[1]: # chose only that category - categ
                continue
            if gold[key] == "neutral" or gold[key] == "absence": # skip neutral and absence category
                continue
            out2write.append(key + "\t" + gold[key] + "\t" + test[key])
            actual.append(gold[key])
            predicted.append(test[key])

        p,r,f,_ = metrics.precision_recall_fscore_support(actual, predicted, average="macro")
        print str(f) + "\t",
        avgs.append(f)
        out2write = [categ + " avg_f1=" + str(f)] + out2write
    print np.average(avgs)
    result_string = "avg_f1="+str(np.average(avgs))

    data_frame = pd.DataFrame({"col":[result_string] + out2write})
    domain = gold_file_name.split("_")[1]
    out_file_name = "eval_Д_"+domain+".csv"
    data_frame.to_csv(out_file_name, index=False, header=False, encoding="utf-8")
    print "see "+out_file_name+" for details"

if __name__ == "__main__":
    main(sys.argv[1:])
    exit()