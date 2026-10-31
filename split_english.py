import importlib.resources
import re

from symspellpy.symspellpy import SymSpell

# Set max_dictionary_edit_distance to avoid spelling correction
sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
dictionary_path = importlib.resources.path("symspellpy", "frequency_dictionary_en_82_765.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
with dictionary_path as dictionary_path:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# a sentence without any spaces
# input_term = "thequickbrownfoxjumpsoverthelazydog"
# input_term = "before15thofeach monthfortheServiceofthe immediatelyprevious monthand subject totimely"
# result = sym_spell.word_segmentation(input_term)
# print(result)
# print(f"{result.corrected_string}, {result.distance_sum}, {result.log_prob_sum}")

NL_PLACEHOLDER = '|||NL|||'


def split_english_words(char_dicts):
    word = ''.join(d['char'] for d in char_dicts)
    if not word or not is_english(word):
        return char_dicts
    protected = word.replace('\n', NL_PLACEHOLDER)
    segmented = sym_spell.word_segmentation(protected).segmented_string
    restored = segmented.replace(NL_PLACEHOLDER, '\n')
    chars_result, i = [], 0
    for ch in restored:
        if i >= len(char_dicts):
            break
        orig = char_dicts[i]
        if orig['char'] == ch:
            chars_result.append(orig)
            i += 1
        else:
            fake = orig.copy()
            fake['char'] = ch
            chars_result.append(fake)

    return chars_result


ENGLISH_RE = re.compile(r'[a-z0-9\s]+', re.IGNORECASE)
def is_english(word: str):
    if not word:
        return False
    removed_word = ENGLISH_RE.sub('', word)
    if len(removed_word) / len(word) < 0.5:
        return True
    return False
