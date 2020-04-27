# coding=utf-8
import numpy as np
import pandas as pd
import sys
from collections import Counter
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

    test_ids = []
    for itm in itemlist:
        review_id = itm.get("id")
        test_ids.append(int(review_id))
        terms = itm.find("aspects").findall("aspect")
        for xml_term in terms:
            if xml_term.get("type")=="explicit" and xml_term.get("mark")=="Rel":
                term_identifier = xml_term.get("from")+"_"+xml_term.get("to")
                category = xml_term.get("category")
                gold[review_id + "_" + term_identifier] = category

    test = {}
    # process file with participant markup
    tree = etree.parse(test_file_name)
    doc = tree.getroot()
    itemlist = doc.findall("review")

    for itm in itemlist:
        review_id = int(itm.get("id"))

        if review_id in test_ids: #it's test review
            terms = itm.find("aspects").findall("aspect")
            for xml_term in terms:
                if xml_term.get("type")=="explicit" and xml_term.get("mark")=="Rel":
                    term_identifier = xml_term.get("from")+"_"+xml_term.get("to")
                    category = xml_term.get("category")
                    test[str(review_id) + "_" + term_identifier] = category
    
    actual = []
    predicted = []
    out2write = ["","id\tactual\tpredicted"]
    for key in gold:
        actual.append(gold[key])
        predicted.append(test[key])
        out2write.append(key + "\t" + gold[key] + "\t" + test[key])

    p,r,f,_ = metrics.precision_recall_fscore_support(actual, predicted, average="macro")
    print "%f\t%f\t%f" % (p,r,f)
    result_string = "avg_f1="+str(f)

    data_frame = pd.DataFrame({"col":[result_string] + out2write})
    domain = gold_file_name.split("_")[1]
    out_file_name = "eval_Г_"+domain+".csv"
    data_frame.to_csv(out_file_name, index=False, header=False, encoding="utf-8")
    print "see "+out_file_name+" for details"

if __name__ == "__main__":
    main(sys.argv[1:])
    exit()