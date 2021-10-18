def compute_scores(cluster, scorer):
    """
    input:
        cluster: list of list of str, each str is sentence.
        scorer: rouge scorer
    output:
        [[{'text': 'I like apple',
            'pegasus_score': 0.3,
            'pyramid_rouge': 0.5},
          {'text': 'And I also like banana',
            'pegasus_score': 0.5,
            'pyramid_rouge': 0.4}],
          ...],
          [{'text': ...,
            'pegasus_score': 0.4,
            'pyramid_rouge': 0.6},
          ...],
        ...]
    """
    # cluster = [[s for p in doc.split('\n') for s in sent_tokenize(p) if s!=''] for doc in documents.split('|||||')]
    result_dict = []
    all_text = "\n".join(["\n".join(d) for d in cluster])
    for i_doc, doc in enumerate(cluster):
        result_dict.append([])
        for i_s, s in enumerate(doc):
            # if s is too short, i.e. less than 5 chars, we directly set scores to 0
            if len(s.split()) < 5:
                result_dict[-1].append(
                    {"text": s, "pegasus_score": 0, "pyramid_rouge": 0}
                )
                continue
            # compute pegasus_score
            ref_sents = all_text.replace(s, "")
            score = compute_rouge_scores(scorer, [s], [ref_sents])
            pegasus_score = score["rouge1"][0].fmeasure

            # compute pyramid rouge score (Cluster ROUGE in the paper)
            pyramid_score = 0
            for i_doc_pyramid, ref in enumerate(cluster):
                if i_doc_pyramid != i_doc:

                    # whole doc version, the rouge scores are computed based on
                    # ROUGE(s_n,doc_m)
                    hyp = [s]
                    ref = [" ".join(ref)]
                    scores = compute_rouge_scores(scorer, hyp, ref)
                    pyramid_score += (
                        scores["rouge1"][0].fmeasure
                        + scores["rouge2"][0].fmeasure
                        + scores["rougeL"][0].fmeasure
                    ) / 3

                else:
                    continue
            result_dict[-1].append(
                {
                    "text": s,
                    "pegasus_score": pegasus_score,
                    "pyramid_rouge": pyramid_score,
                }
            )
    return result_dict


def compute_rouge_scores(scorer, predictions, references):
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    assert len(predictions) == len(references)
    all_scores = []
    for pred, ref in zip(predictions, references):
        all_scores.append(scorer.score(target=ref, prediction=pred))
    final_scores = {}
    for score_type in all_scores[0].keys():
        final_scores[score_type] = [
            all_scores[i][score_type] for i in range(len(all_scores))
        ]
    return final_scores


def get_entities(nlp, all_docs):
    all_entities_pyramid = {}
    all_entities = {}
    for i, doc in enumerate(all_docs):
        all_entities_cur = set()
        for s in doc:
            sent = nlp(s["text"])
            if len(sent.ents) != 0:
                for ent in sent.ents:
                    all_entities_cur.add(ent.text)
                    all_entities[ent.text] = all_entities.get(ent.text, 0) + 1
        for e in all_entities_cur:
            all_entities_pyramid[e] = all_entities_pyramid.get(e, 0) + 1
    return all_entities_pyramid, all_entities
