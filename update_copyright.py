# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
import sys
import re
import os
from github import Github
import datetime
import fnmatch
import argparse

comment_types = {
    "//": ['.java', '.cpp', '.c', '.rs'],
    "#": ['.py']
}

skip_files = [
    "*.md",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile*"
]

hb_users_outside_organization = [""]

current_year = datetime.datetime.now().year
copyright_header_regex = f"Copyright \(C\) {current_year} Habana Labs, Ltd. an Intel Company.$"
copyright_header = f"Copyright (C) {current_year} Habana Labs, Ltd. an Intel Company."


def create_github_instance(token):
    try:
        github_instance = Github(token)
        return github_instance
    except Exception as e:
        print("Error:", e)
        return None


def match_comment(type):
    for comment, extensions in comment_types.items():
        if type in extensions:
            return comment
    return None


def get_merged_pull_request_numbers(github_instance, repository_name, organization_name, branch_name):
    try:
        repo = github_instance.get_repo(repository_name)
        pull_requests = repo.get_pulls(
            state='closed', sort='updated', direction='desc')
        members = [member.login for member in github_instance.get_organization(
            organization_name).get_members()]
        commits = set(
            [commit.sha for commit in repo.get_commits(sha=branch_name)])
        pr_numbers = []
        rejested_users = set()
        for pr in pull_requests:
            if pr.merged and (pr.user.login in members or pr.user.login in hb_users_outside_organization):
                if pr.merge_commit_sha and commits:
                    pr_numbers.append(pr.number)
            else:
                rejested_users.add(pr.user.login)
        return pr_numbers, rejested_users
    except Exception as e:
        print("Error fetching merged pull request numbers:", e)
        return [], []


def check_and_update_copyright_header(file_path):
    try:
        for ignored_path in skip_files:
            if fnmatch.fnmatch(os.path.basename(file_path), ignored_path):
                return
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            extension = os.path.splitext(file_path)[1].lower()
            comment = match_comment(extension)
            if comment is None:
                print("Unknown extension: ", file_path)
                return
            match = re.search(copyright_header_regex,
                              file_content, re.MULTILINE)
            if not match:
                updated_content = comment + " " + copyright_header + "\n"
                updated_content += file_content
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(updated_content)
                    print("Added missing copyright header to:", file_path)
    except Exception as e:
        print("Error updating copyright header in file:", file_path, e)


def main(args):
    github_token = args.github_token
    repository_name = args.repository_name
    organization_name = args.organization_name
    branch_name = args.branch_name

    github_instance = create_github_instance(github_token)

    if github_instance:
        pr_numbers, rejected_users = get_merged_pull_request_numbers(
            github_instance, repository_name, organization_name, branch_name)
        if pr_numbers:
            for pr_number in pr_numbers:
                pr = github_instance.get_repo(
                    repository_name).get_pull(pr_number)
                files = pr.get_files()
                for file in files:
                    check_and_update_copyright_header(file.filename)
        else:
            print("No merged pull requests found for the specified repository '{}' by members of organization '{}' merged into branch '{}'".format(
                repository_name, organization_name, branch_name))

        if rejected_users:
            print("Users rejected due to unknown authenticity:")
            print(rejected_users)
    else:
        print("Failed to create GitHub instance. Please check your token.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub script")
    parser.add_argument("--github_token", type=str, help="GitHub token")
    parser.add_argument("--repository_name", type=str, help="Repository name")
    parser.add_argument("--organization_name", type=str,
                        help="Organization name")
    parser.add_argument("--branch_name", type=str, help="Branch name")
    args = parser.parse_args()
    main(args)
