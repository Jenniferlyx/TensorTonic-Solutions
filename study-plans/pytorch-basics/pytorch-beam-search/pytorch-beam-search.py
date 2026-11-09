def beam_search(log_probs_fn, start_token: int, end_token: int, beam_width: int, max_len: int) -> list[int]:
    """
    Returns token IDs including the start token but excluding a terminal end token.
    """
    beams = [([start_token], 0.0)]
    complete = []
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == end_token:
                complete.append((seq[:-1], score))
                continue
            probs = log_probs_fn(seq)
            for token_id, log_p in enumerate(probs):
                candidates.append((seq + [token_id], score + log_p))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
    all_seq = complete + beams
    all_seq.sort(key=lambda x: x[1], reverse=True)
    result = all_seq[0][0]
    if result and result[-1] == end_token:
        result = result[:-1]
    return result
            
