#!/bin/sh
# prepare files for distribution after release
# Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
# Date: 2022-09-26

OUTDIR=$1

## assert that the output dir is the VGG software page
if [ ! -d $OUTDIR/downloads ]; then
    echo "The output folder should contain a downloads folder"
    exit
fi

VIA_SCRIPT_DIR=`dirname "$(realpath $0)"`
VIA_CODE_DIR="${VIA_SCRIPT_DIR}/../"
VIA_VERSION=`cat ${VIA_CODE_DIR}/src/via.js | grep "var VIA_VERSION" | cut -d "=" -f 2 | cut -d "'" -f2`

echo $VIA_CODE_DIR
echo $VIA_VERSION

$VIA_SCRIPT_DIR/pack_all.sh

## via-2.0.12.zip
VIA_DIR="${OUTDIR}/downloads/via-${VIA_VERSION}"
if [ ! -d $VIA_DIR ]; then
	mkdir $VIA_DIR
fi
cp "${VIA_CODE_DIR}/dist/via.html" "${VIA_DIR}/via.html"
cp "${VIA_CODE_DIR}/dist/via_demo.html" "${VIA_DIR}/via_demo.html"
cp "${VIA_CODE_DIR}/README.md" "${VIA_DIR}/README.md"
cp "${VIA_CODE_DIR}/CHANGELOG" "${VIA_DIR}/CHANGELOG"
cp "${VIA_CODE_DIR}/Contributors.md" "${VIA_DIR}/Contributors.md"
cp "${VIA_CODE_DIR}/LICENSE" "${VIA_DIR}/LICENSE"

## via-src-2.0.12.zip
VIA_SRCDIR="${OUTDIR}/downloads/via-src-${VIA_VERSION}"
if [ ! -d $VIA_SRCDIR ]; then
	mkdir $VIA_SRCDIR
fi
cp -fr "${VIA_CODE_DIR}/" "${VIA_SRCDIR}/"

## create zip
find "${VIA_DIR}" -type f -iname '.*' -delete
find "${VIA_DIR}" -type f -iname '*.*~' -delete
find "${VIA_SRCDIR}" -type f -iname '.*' -delete
find "${VIA_SRCDIR}" -type f -iname '*.*~' -delete

cd "${OUTDIR}/downloads/"
zip -r "${OUTDIR}/downloads/via-${VIA_VERSION}.zip" "${VIA_DIR}"
cd "${OUTDIR}/downloads/"
zip -r "${OUTDIR}/downloads/via-src-${VIA_VERSION}.zip" "${VIA_SRCDIR}"

#rm -fr "${VIA_DIR}"
#rm -fr "${VIA_SRCDIR}"