CREATE OR REPLACE FUNCTION get_feature_phrases_trans
(
    state DOUBLE PRECISION[],
    qvec  DOUBLE PRECISION[],
    pvec  DOUBLE PRECISION[],
    senti DOUBLE PRECISION,
    count INTEGER
) 
RETURNS DOUBLE PRECISION[] AS $$
    def norm(vec):
        norm = 0.
        for v in vec:
            norm += v*v
        norm = norm**(1./2.)
        return norm

    def cosine(vec1, vec2):
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        if norm1 > 0 and norm2 > 0 and len(vec1) == len(vec2):
            dot = 0.
            for i in range(len(vec1)):
                dot += vec1[i] * vec2[i]
            return dot / norm1 / norm2
        else:
            return 0.

    if state[0] == 0: # first row
        sum_phrases = [0] * 300
    else:
        sum_phrases = state[10:]
    num_rows = state[0] + 1
    sim_count = state[1]
    count2 = state[2]
    sent_sum = state[3]
    sent_sum2 = state[4]
    pos_count = state[5]
    neg_count = state[6]
    pos_match_count = state[7]
    neg_match_count = state[8]

    for i, v in enumerate(pvec):
        sum_phrases[i] += v * count

    if cosine(qvec, pvec) > 0.8:
        sim_count += count
        sent_sum += count * senti
        if senti >= 0:
            pos_match_count += count
        else:
            neg_match_count += count

    sent_sum2 += count * senti
    count2 += count
    if senti >= 0:
        pos_count += count
    else:
        neg_count += count

    return [num_rows, sim_count, count2, sent_sum, sent_sum2, pos_count, neg_count, pos_match_count, neg_match_count, cosine(qvec, sum_phrases)] + sum_phrases
$$ language 'plpython3u';

CREATE OR REPLACE FUNCTION get_feature_phrases_final(state DOUBLE PRECISION[])
RETURNS DOUBLE PRECISION[] AS $$
    num_rows = state[0]
    sim_count = state[1]
    count2 = state[2]
    sent_sum = state[3]
    sent_sum2 = state[4]
    pos_count = state[5]
    neg_count = state[6]
    pos_match_count = state[7]
    neg_match_count = state[8]
    sum_phrases_cosine = state[9]

    ret =  [sim_count, sent_sum / sim_count, sent_sum2 / count2, pos_count, neg_count, pos_match_count, neg_match_count, sum_phrases_cosine]
    plpy.info(str(ret))
    return ret
$$ language 'plpython3u';

DROP AGGREGATE IF EXISTS get_feature_phrases(DOUBLE PRECISION [],DOUBLE PRECISION [], DOUBLE PRECISION, INTEGER);
CREATE AGGREGATE get_feature_phrases(
    DOUBLE PRECISION[],
    DOUBLE PRECISION[],
    DOUBLE PRECISION,
    INTEGER
) (
    STYPE=DOUBLE PRECISION[],
    SFUNC=get_feature_phrases_trans,
    FINALFUNC=get_feature_phrases_final,
    INITCOND='{0,1,1,0,0,0,0,0,0}'
);

