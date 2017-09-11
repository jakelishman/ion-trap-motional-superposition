#!/bin/bash

# SYNOPSIS
#   submit.sh [options]
#
# OPTIONS
#   -d
#       Debug run.  This will print out more information about the execution of
#       the script, and won't submit the job to qsub at the end.
#
#   -b before (default = 3)
#       Each colour sequence will be tried up to "before" times (with different
#       randomised initial inputs) searching for the first success.  If no
#       successful optimisation occurs in those attempts, the colour sequence
#       will be marked a failure.
#
#   -a after (default = 4)
#       After the first success of a colour sequence, the same sequence will be
#       tried "after" more times to try and find other angle sequences.  All
#       successful angle sequences will be reported.
#
#    -i input_file (default = stdin)
#       Read the targets file from "input_file" rather than from the default of
#       stdin.  If multiple '-i' flags are given, the files will be read as if
#       they had been concatenated in the order they're specified.
#
#   -o outdir (default = .)
#       The directory to place the output files from the run in.  Assumed to be
#       "." if not supplied.
#
#   -w walltime (default = 24:00:00)
#       The amount of walltime to request from the scheduler.

# Command-line option initialisation.
opts_debug=
opts_before=
opts_after=
opts_input_files=
opts_output_dir=
opts_walltime=

# Fill in any options which need to be filled to continue.
fill_missing_parameters() {
    if [[ -z $opts_debug ]]; then opts_debug=false; fi
    if [[ -z $opts_before ]]; then opts_before=3; fi
    if [[ -z $opts_after ]]; then opts_after=4; fi
    # We don't set the input files because it will be stdin by default.
    if [[ -z $opts_output_dir ]]; then opts_output_dir="$(pwd -P)"; fi
    if [[ -z $opts_walltime ]]; then opts_walltime=24:00:00; fi
}

# Separator character for file names.
file_sep=";"

# Output files that will be written.
output_inputs_file="inputs_file"
output_log_file="message_log"
output_results_file="results"
output_parameters_file="parameters"
output_submission_file="submission"

# Try to find the directory with the script in...
code_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
# ...so we can find the python files.
if [[ ! -e "$code_dir/find_sequences.py" ]]; then
    echo "ERROR: couldn't find 'find_sequences.py'." >&2
    echo "       It should be in the same directory as this script." >&2
    exit 1
fi

# Exit the script with an error message if a variable is already set.
#
# Usage: fail_if_set string name
#   string: string to test if it has content.
#   name: name of the flag which this string is associated with.
fail_if_set() {
    if [[ ! -z "$1" ]]; then
        echo "ERROR: the '$2' flag has been set multiple times." >&2
        exit 1
    fi
}

# Exit the script with an error message if there is no argument.
#
# Usage: fail_if_no_arg string name
#   string: string to test if empty.
#   name: name of flag which this string is associated with.
fail_if_no_arg() {
    if [[ -z "$1" ]]; then
        echo "ERROR: you must supply an argument to the '$2' flag." >&2
        exit 1
    fi
}

# Exit the script if an input file couldn't be found.
#
# Usage: fail_if_file_does_not_exist file
#   file: file to test
fail_if_file_does_not_exist() {
    if [[ ! -e "$1" ]]; then
        echo "ERROR: can't find the file '$1'." >&2
        exit 1
    fi
}

# Check that the given directory has no files which will conflict with the
# outputs.
#
# Usage: check_output_directory_is_safe dir
#   dir: directory to test
check_output_directory_is_safe() {
    for file in "$output_inputs_file" "$output_log_file" "$output_results_file"\
             "$output_parameters_file" "$output_submission_file"; do
        if [[ -e "$1/$file" ]]; then
            echo "ERROR: output directory '$1' has file conflicts." >&2
            exit 1
        fi
    done
}

# Print out the parameter list to standard output.  Takes no arguments.
print_parameters() {
    echo "\$opts_debug       : '$opts_debug'"
    echo "\$opts_before      : '$opts_before'"
    echo "\$opts_after       : '$opts_after'"
    echo "\$opts_walltime    : '$opts_walltime'"
    echo "\$opts_output_dir  : '$opts_output_dir'"
    echo -n "\$opts_input_files :"
    if [[ -z "${opts_input_files[0]}" ]]; then
        echo " ''"
    else
        echo " '{"
        for file in "${opts_input_files[@]}"; do
            echo "    \"$file\","
        done
        echo "}'"
    fi
}

# Parse the command line arguments.
while (($#)); do
    case "$1" in
        "-d" | "--debug")
            opts_debug=true ;;

        "-b" | "--before")
            fail_if_set "$opts_before" "$1"
            fail_if_no_arg "$2" "$1"
            opts_before=$2
            shift ;;

        "-b"[0-9]*)
            fail_if_set "$opts_before" "-b"
            opts_before=${1:2} ;;

        "-a" | "--after")
            fail_if_set "$opts_after" "$1"
            fail_if_no_arg "$2" "$1"
            opts_after="$2"
            shift ;;

        "-a"[0-9]*)
            fail_if_set "$opts_after" "-b"
            opts_before=${1:2} ;;

        "-i" | "--input")
            fail_if_no_arg "$2" "$1"
            fail_if_file_does_not_exist "$2"
            if [[ -z $opts_input_files ]]; then
                opts_input_files="$(realpath "$2")"
            else
                opts_input_files+="${file_sep}$(realpath "$2")"
            fi
            shift;;

        "-o" | "--output")
            fail_if_set "$opts_output_dir" "$1"
            fail_if_no_arg "$2" "$1"
            # Create the output directory if necessary.
            mkdir -p "$2" 2>/dev/null
            if [[ ! $? = 0 ]]; then
                echo "Couldn't make the output directory '$2'" >&2
                exit 1
            fi
            check_output_directory_is_safe "$2"
            opts_output_dir="$(realpath "$2")"
            shift ;;

        "-w" | "--walltime")
            fail_if_set "$opts_walltime" "$1"
            fail_if_no_arg "$2" "$1"
            opts_walltime="$2"
            shift ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            exit 1 ;;
    esac

    shift
done

# Split the ':' separated string of filenames into an array of files.
IFS="${file_sep}" read -r -a opts_input_files <<< "$opts_input_files"

if [[ $opts_debug = true ]]; then
    echo "After reading command line options, I have" >&2
    print_parameters >&2
fi

fill_missing_parameters

if [[ $opts_debug = true ]]; then
    echo >&2
    echo "After filling in any necessary missing parameters, I have" >&2
    print_parameters >&2
fi

# Change into the output directory for making the files.
cd "$opts_output_dir"

echo "Time submitted: $(date +"%F %T")" > "$output_parameters_file"
print_parameters >> "$output_parameters_file"

# Save the input parameters into the relevant file.
if [[ -z "$opts_input_files" ]]; then
    echo '# [stdin]' > "$output_inputs_file"
    cat >> "$output_inputs_file"
    echo >> "$output_inputs_file"
else
    rm -f "$output_inputs_file"
    for input in "${opts_input_files[@]}"; do
        echo "# ${input}" >> "$output_inputs_file"
        cat "${input}" >> "$output_inputs_file"
        echo >> "$output_inputs_file"
    done
fi

# Write out the script into the right place.
cat > ${output_submission_file} << ScriptEnd
#!/bin/bash
#PBS -l walltime=${opts_walltime}
#PBS -l select=1:ncpus=1:mem=128mb

cp "${code_dir}"/*.py "${opts_output_dir}/${output_inputs_file}" .

# Terminate python 10 minutes before the walltime, so we can catch any results
# instead of overrunning and losing everything.  I use my own copy of it with a
# shorter delay, so test runs don't take so long.
#
# The eval is necessary to get correct input and output redirection inside the
# pbsexec script.
${HOME}/bin/pbsexec.sh -grace 10\
    'eval python -O find_sequences.py ${opts_before} ${opts_after}\
        <"${output_inputs_file}"\
        >"${output_results_file}"\
        2>"${output_log_file}"'

cp "${output_results_file}" "${output_log_file}" "${opts_output_dir}"/
ScriptEnd

if [[ "$opts_debug" = false ]]; then
    qsub "${output_submission_file}"
fi
