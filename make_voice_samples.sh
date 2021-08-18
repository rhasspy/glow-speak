#!/usr/bin/env bash
if [[ -z "$2" ]]; then
    echo "Usage: make_voice_samples.sh <VOICES_DIR> <OUTPUT_DIR> [MODEL]..."
    exit 1
fi

voices_dir="$1"
output_dir="$2"
shift 2  # Remaining arguments will be model names

mkdir -p "${output_dir}"

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"

# Avoid bash built-in "time" command that doesn't allow output to a file
time_prog="$(which time)"

# -----------------------------------------------------------------------------

# Pangrams from: http://clagnut.com/blog/2380/
declare -A pangrams
pangrams['cmn']='我的氣墊船裝滿了鱔魚 你要不要跟我跳舞？'
pangrams['en-us']='The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as young Arthur wanted.'
pangrams['es']='Benjamín pidió una bebida de kiwi y fresa; Noé, sin vergüenza, la más exquisita champaña del menú.'
pangrams['de']='Falsches Üben von Xylophonmusik quält jeden größeren Zwerg.'
pangrams['el']='Ταχίστη αλώπηξ βαφής ψημένη γη, δρασκελίζει υπέρ νωθρού κυνός'
pangrams['fi']='Fahrenheit ja Celsius yrjösivät Åsan backgammon-peliin, Volkswagenissa, daiquirin ja ZX81:n yhteisvaikutuksesta.'
pangrams['fr']='Buvez de ce whisky que le patron juge fameux.'
pangrams['hu']='Jó foxim és don Quijote húszwattos lámpánál ülve egy pár bűvös cipőt készít.'
pangrams['it']='Ma la volpe, col suo balzo, ha raggiunto il quieto Fido.'
pangrams['ko']='키스의 고유조건은 입술끼리 만나야 하고 특별한 기술은 필요치 않다.'
pangrams['nl']='Pa’s wijze lynx bezag vroom het fikse aquaduct.'
pangrams['ru']='Широкая электрификация южных губерний даст мощный толчок подъёму сельского хозяйства.'
pangrams['sv']='Yxskaftbud, ge vår WC-zonmö IQ-hjälp.'
pangrams['sw']='Gari langu linaloangama limejaa na mikunga. Nakutakia siku njema!'
pangrams['vi']='Bạn có nói tiếng Việt không? Một thứ tiếng thì không bao giờ đủ.'

# -----------------------------------------------------------------------------

function test_model {
    model_path="$1"
    model_lang="$(cat "${model_path}/LANGUAGE")"
    model_name="$(basename "${model_path}")"

    # Get pangram for this language
    pangram="${pangrams["${model_lang}"]}"
    if [[ -z "${pangram}" ]]; then
        echo "WARNING: No pangram available for language ${model_lang}"
        return
    fi

    # Write WAV and two text files, one with timing info and the other with stdout/stderr
    sample_prefix="${model_name}"
    sample_wav="${output_dir}/${sample_prefix}.wav"
    sample_time="${output_dir}/${sample_prefix}_time.txt"
    sample_error="${output_dir}/${sample_prefix}_output.txt"

    # Time TTS command
    echo "${model_path}"
    "${time_prog}" "--output=${sample_time}" -- \
                   "${this_dir}/bin/glow-speak" --tts "${model_path}" --quality high --output-file "${sample_wav}" -- "${pangram}" 2>&1 \
        | tee "${sample_error}"

}

# -----------------------------------------------------------------------------

if [[ -z "$1" ]]; then
    # All models from voices_dir
    while read -r model_path; do
        test_model "${model_path}"
    done < <(find "${voices_dir}" -maxdepth 1 -mindepth 1 -type d)
else
    # Specific model(s)
    while [[ -n "$1" ]]; do
        test_model "$1"
        shift 1
    done
fi
