# coding=utf-8
import sys
import optparse
from lxml import etree

def get_terms(item, gold, task):
    units = []
    review_id = int(item.get("id"))
    content = item.find("text").text
    if gold:
        terms = item.find("aspects").findall("aspect")
    else:
        if item.find("aspects1") == None:
            return review_id, 0, units
        terms = item.find("aspects1").findall("aspect")
    terms_count = 0

    term_set = [] # we don't have to take repeated terms
    
    for xml_term in terms:
        if xml_term.get("mark")=="Rel":
            if task == "a": #task switch
                if xml_term.get("type")!="explicit":
                    continue

            term_identifier = xml_term.get("from")+"_"+xml_term.get("to")
            if term_identifier in term_set:
                continue
            term_set.append(term_identifier)
            
            terms_count += 1

            written_term = xml_term.get("term")
            position_from = int(xml_term.get("from"))
            position_to = int(xml_term.get("to"))
            term = content[position_from:position_to]
            #if written_term != term:
            #    print review_id, "terms does't match [", str(position_from), str(position_to), ") ->", term, "<>", written_term

            units.append(str(position_from)+'_'+str(position_to))
    
    return review_id, terms_count, units

def get_units(item, gold, task):
    units = []
    review_id = int(item.get("id"))
    content = item.find("text").text
    if gold:
        terms = item.find("aspects").findall("aspect")
    else:
        if item.find("aspects1") == None:
            return review_id, 0, units
        terms = item.find("aspects1").findall("aspect")

    terms_count = 0

    term_set = [] # we don't have to take repeated terms

    for xml_term in terms:
        if xml_term.get("mark")=="Rel":
            if task == "a": #task switch
                if xml_term.get("type")!="explicit":
                    continue

            term_identifier = xml_term.get("from")+"_"+xml_term.get("to")
            
            if term_identifier in term_set:
                continue
            term_set.append(term_identifier)

            terms_count += 1

            written_term = xml_term.get("term")
            position_from = int(xml_term.get("from"))
            position_to = int(xml_term.get("to"))
            term = content[position_from:position_to]
            #if written_term != term:
            #    print review_id, "terms does't match [", str(position_from), str(position_to), ") ->", term, "<>", written_term

            start = position_from
            for i, unit in enumerate(term.split(' ')):
                end = start + len(unit)
                units.append(str(start)+'_'+str(end))
                start = end + 1
    
    return review_id, terms_count, units

def main(argv=None):
    # parse the input
    parser = optparse.OptionParser()
    parser.add_option('-g')
    parser.add_option('-t')
    parser.add_option('-a')
    parser.add_option('-w')
    options, args = parser.parse_args()
    gold_file_name = options.g
    test_file_name = options.t
    task = options.a
    alg_type = options.w

    # process file with gold markup
    tree = etree.parse(gold_file_name)
    doc = tree.getroot()
    itemlist = doc.findall("review")

    idx2units = {}
    for itm in itemlist:
        if alg_type == "weak":
            idx, terms_count, units = get_units(itm, True, task)
        else:
            idx, terms_count, units = get_terms(itm, True, task)

        idx2units[idx] = (terms_count, units)

    # process file with participant markup
    tree = etree.parse(test_file_name)
    doc = tree.getroot()
    itemlist = doc.findall("review")

    print "id\tcorrect_unit_count\textracted_unit_coun\tmatch_count\tp\tr\tf"
    total_p, total_r, total_f = .0, .0, .0
    processed = []
    for itm in itemlist:
        idx = int(itm.get("id"))

        if alg_type == "weak":
            idx, terms_count, units = get_units(itm, False, task)
        else:
            idx, terms_count, units = get_terms(itm, False, task)

        if idx in idx2units and not idx in processed: #it's not processed test review
            processed.append(idx)
            correct = idx2units[idx][1]
            correct4del = [i for i in correct]
            extracted = units
            match = []
            for i in extracted:
                if i in correct4del:
                    match.append(i)
                    correct4del.remove(i)

            r = float(len(match))/len(correct)
            p = float(len(match))/len(extracted) if len(extracted) != 0 else 0
            if p == 0 and r == 0:
                f = 0
            else:
                f = (2*p*r)/(p+r)
            total_p += p
            total_r += r
            total_f += f
            print "%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f" % (idx, len(correct),len(extracted), len(match), p, r, f)

    n = len(idx2units.keys())
    print "%f\t%f\t%f" % (total_p/n, total_r/n, total_f/n)
    
if __name__ == "__main__":
    main(sys.argv[1:])
    exit()