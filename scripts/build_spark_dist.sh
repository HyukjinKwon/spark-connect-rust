#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Builds the Spark Connect distribution of a commit of Apache Spark, or of a fork of it, the
# layout of the released spark-X.Y.Z-bin-hadoop3-connect package, which the jobs that download a
# release start their server from:
#
#   scripts/build_spark_dist.sh [REPOSITORY [REF]] DIR
#
# REPOSITORY is a GitHub repository (apache/spark, HyukjinKwon/spark), a git URL or the path of a
# local repository (whose unpushed commits it can build), and REF a branch, a tag or a commit. They
# default to the commit that the vendored protos are of (crates/spark-connect-proto/proto/
# SPARK_REPO.txt and SPARK_SHA.txt), so that the server speaks the client's protocol. The
# distribution is unpacked into DIR/spark-<version>-bin-ci-connect, whose path the script prints
# last; DIR/SPARK_COMMIT records the commit it was built of.
#
# It builds with sbt (dev/make-distribution.sh --sbt-enabled), with -Phadoop-3 -Phive
# -Phive-thriftserver as the release builds the hadoop3 package. It needs a JDK that the commit
# builds with (21 for Spark 5.0), git and a network that resolves Maven Central (set
# MAVEN_MIRROR_URL for a mirror). The checkout goes to SPARK_SRC (default DIR/src), and is kept,
# so that a second run of the same commit only packages it again.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
proto="${root}/crates/spark-connect-proto/proto"
case "$#" in
  1) repo=""; ref=""; dir=$1 ;;
  2) repo=$1; ref=""; dir=$2 ;;
  3) repo=$1; ref=$2; dir=$3 ;;
  *) echo "usage: $0 [REPOSITORY [REF]] DIR" >&2; exit 2 ;;
esac
[ -n "${repo}" ] || repo="$(tr -d '[:space:]' < "${proto}/SPARK_REPO.txt")"
[ -n "${ref}" ] || ref="$(tr -d '[:space:]' < "${proto}/SPARK_SHA.txt")"
case "${repo}" in
  http*://* | git@* | /* | file://*) url="${repo}" ;;
  *) url="https://github.com/${repo}.git" ;;
esac

if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
  sha="${ref}"
elif [ -d "${url}" ]; then
  # A local repository resolves any revision, such as an abbreviated commit that is not pushed.
  sha="$(git -C "${url}" rev-parse --verify --quiet "${ref}^{commit}" || true)"
else
  sha="$(git ls-remote "${url}" "${ref}" "refs/heads/${ref}" "refs/tags/${ref}" \
    | awk 'NR == 1 {print $1}')"
fi
if [ -z "${sha}" ]; then
  echo "::error::${ref} is not a revision of ${url}" >&2
  exit 1
fi
echo "Building the Spark Connect distribution of ${url} ${sha}"

mkdir -p "${dir}"
dir="$(cd "${dir}" && pwd)"
src="${SPARK_SRC:-${dir}/src}"
# SPARK_SRC can be a checkout, or a worktree (whose .git is a file), of the commit already.
if [ ! -e "${src}/.git" ]; then
  git init --quiet "${src}"
  git -C "${src}" remote add origin "${url}"
fi
if [ "$(git -C "${src}" rev-parse -q --verify HEAD 2>/dev/null)" != "${sha}" ]; then
  git -C "${src}" fetch --quiet --depth 1 origin "${sha}"
  git -C "${src}" checkout --quiet --force FETCH_HEAD
fi

cd "${src}"
rm -f spark-*-bin-ci-connect.tgz spark-*-bin-ci.tgz
./dev/make-distribution.sh --name ci --tgz --connect --sbt-enabled \
  -Phadoop-3 -Phive -Phive-thriftserver
tgz="$(ls spark-*-bin-ci-connect.tgz)"
rm -rf "${dir}"/spark-*-bin-ci-connect
tar -xzf "${tgz}" -C "${dir}"
echo "${url} ${sha}" > "${dir}/SPARK_COMMIT"
ls -d "${dir}/${tgz%.tgz}"
