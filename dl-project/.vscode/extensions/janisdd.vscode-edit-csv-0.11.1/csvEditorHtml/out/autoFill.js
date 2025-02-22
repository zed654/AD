"use strict";
const intRegex = /-?\d+/g;
const floatRegexEn = /-?\d+\.\d+/g;
const floatRegexNonEn = /-?\d+\,\d+/g;
const monthRegexLen3 = /^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$/gi;
const monthRegexFull = /^(january|february|march|april|may|june|july|august|september|october|november|december)$/gi;
const monthFullNames = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
const monthShortNames = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
const allKnownSeparatorRegex = /[\-\/\.]/g;
const allKnownDateFormats = new Map();
allKnownDateFormats.set(`YYYY-MM-DD`, {
    regex: /^(\d{4})[\-\/\.](\d{2})[\-\/\.](\d{2})$/g,
});
allKnownDateFormats.set(`YYYY-M-DD`, {
    regex: /^(\d{4})[\-\/\.](\d{1})[\-\/\.](\d{2})$/g,
});
allKnownDateFormats.set(`YYYY-MM-D`, {
    regex: /^(\d{4})[\-\/\.](\d{2})[\-\/\.](\d{1})$/g,
});
allKnownDateFormats.set(`YYYY-M-D`, {
    regex: /^(\d{4})[\-\/\.](\d{1})[\-\/\.](\d{1})$/g,
});
allKnownDateFormats.set(`DD-MM-YYYY`, {
    regex: /^(\d{2})[\-\/\.](\d{2})[\-\/\.](\d{4})$/g,
});
allKnownDateFormats.set(`DD-M-YYYY`, {
    regex: /^(\d{2})[\-\/\.](\d{1})[\-\/\.](\d{4})$/g,
});
allKnownDateFormats.set(`D-MM-YYYY`, {
    regex: /^(\d{1})[\-\/\.](\d{2})[\-\/\.](\d{4})$/g,
});
allKnownDateFormats.set(`D-M-YYYY`, {
    regex: /^(\d{1})[\-\/\.](\d{1})[\-\/\.](\d{4})$/g,
});
allKnownDateFormats.set(`DD-MM-YY`, {
    regex: /^(\d{2})[\-\/\.](\d{2})[\-\/\.](\d{2})$/g,
});
allKnownDateFormats.set(`DD-M-YY`, {
    regex: /^(\d{2})[\-\/\.](\d{1})[\-\/\.](\d{2})$/g,
});
allKnownDateFormats.set(`D-MM-YY`, {
    regex: /^(\d{1})[\-\/\.](\d{2})[\-\/\.](\d{2})$/g,
});
allKnownDateFormats.set(`D-M-YY`, {
    regex: /^(\d{1})[\-\/\.](\d{1})[\-\/\.](\d{2})$/g,
});
function customAutoFillFunc(_data, targetCount, isNormalDirection, mouseupEvent) {
    if (mouseupEvent.altKey) {
        return [];
    }
    if (!isNormalDirection) {
        _data = [..._data].reverse();
    }
    let numbersStyleToUse = getNumbersStyleFromUi();
    let groupInterpolationInfos = [];
    for (let i = 0; i < _data.length; i++) {
        const cellText = _data[i];
        {
            let checkOtherCases = true;
            const matchesInt = Array.from(cellText.matchAll(intRegex));
            let firstMatchInt = matchesInt.length > 0 ? matchesInt[0] : null;
            let startsWithNumberInt = matchesInt.length > 0 && firstMatchInt && firstMatchInt.index === 0;
            let lastMatchInt = matchesInt.length > 0 ? matchesInt[matchesInt.length - 1] : null;
            let endsWithNumberInt = matchesInt.length > 0 && lastMatchInt && lastMatchInt.index + lastMatchInt[0].length === cellText.length;
            let onlyNumberInt = startsWithNumberInt && endsWithNumberInt && matchesInt.length === 1;
            let matchesFloat = [];
            if (numbersStyleToUse.key === 'en') {
                matchesFloat = Array.from(cellText.matchAll(floatRegexEn));
            }
            else {
                matchesFloat = Array.from(cellText.matchAll(floatRegexNonEn));
            }
            let firstMatchFloat = matchesFloat.length > 0 ? matchesFloat[0] : null;
            let startsWithNumberFloat = matchesFloat.length > 0 && firstMatchFloat && firstMatchFloat.index === 0;
            let lastMatchFloat = matchesFloat.length > 0 ? matchesFloat[matchesFloat.length - 1] : null;
            let endsWithNumberFloat = matchesFloat.length > 0 && lastMatchFloat && lastMatchFloat.index + lastMatchFloat[0].length === cellText.length;
            let onlyNumberFloat = startsWithNumberFloat && endsWithNumberFloat && matchesFloat.length === 1;
            if (onlyNumberFloat) {
                let groupInterpolationInfo_Int = {
                    type: 'int',
                };
                groupInterpolationInfos.push(groupInterpolationInfo_Int);
                checkOtherCases = false;
            }
            else if (onlyNumberInt) {
                let groupInterpolationInfo_Int = {
                    type: 'int',
                };
                groupInterpolationInfos.push(groupInterpolationInfo_Int);
                checkOtherCases = false;
            }
            else if (startsWithNumberInt && endsWithNumberInt) {
                for (const [format, knownFormat] of allKnownDateFormats) {
                    const dateMatches = Array.from(cellText.matchAll(knownFormat.regex));
                    if (dateMatches.length != 1)
                        continue;
                    let dateMatch = dateMatches[0];
                    let dateMatchString = dateMatch[0];
                    const separatorMatches = Array.from(dateMatchString.matchAll(allKnownSeparatorRegex));
                    if (separatorMatches.length !== 2)
                        continue;
                    let separator1 = separatorMatches[0][0];
                    let separator2 = separatorMatches[1][0];
                    let dashIndex1 = format.indexOf('-');
                    let dashIndex2 = format.indexOf('-', dashIndex1 + 1);
                    let displayFormat = format.substring(0, dashIndex1) + separator1 + format.substring(dashIndex1 + 1, dashIndex2) + separator2 + format.substring(dashIndex2 + 1);
                    let originalDate = dayjs(dateMatchString, displayFormat, true);
                    if (originalDate.isValid() === false)
                        continue;
                    let groupInterpolationInfo_Date = {
                        type: 'date',
                        originalDate,
                        displayFormat: displayFormat,
                        separator1String: separator1,
                        separator2String: separator2,
                    };
                    groupInterpolationInfos.push(groupInterpolationInfo_Date);
                    checkOtherCases = false;
                    break;
                }
            }
            const monthNames = Array.from(cellText.matchAll(monthRegexLen3));
            const monthNamesFull = Array.from(cellText.matchAll(monthRegexFull));
            if (checkOtherCases) {
                if (startsWithNumberFloat && firstMatchFloat) {
                    let groupInterpolationInfo_ContainsNumber = {
                        type: 'containsInt',
                        startIndexNumber: firstMatchFloat.index,
                        endIndexNumber: firstMatchFloat.index + firstMatchFloat[0].length,
                        numberString: firstMatchFloat[0]
                    };
                    groupInterpolationInfos.push(groupInterpolationInfo_ContainsNumber);
                }
                else if (endsWithNumberFloat && lastMatchFloat) {
                    let groupInterpolationInfo_ContainsNumber = {
                        type: 'containsInt',
                        startIndexNumber: lastMatchFloat.index,
                        endIndexNumber: lastMatchFloat.index + lastMatchFloat[0].length,
                        numberString: lastMatchFloat[0]
                    };
                    groupInterpolationInfos.push(groupInterpolationInfo_ContainsNumber);
                }
                else if (startsWithNumberInt && firstMatchInt) {
                    let groupInterpolationInfo_ContainsNumber = {
                        type: 'containsInt',
                        startIndexNumber: firstMatchInt.index,
                        endIndexNumber: firstMatchInt.index + firstMatchInt[0].length,
                        numberString: firstMatchInt[0]
                    };
                    groupInterpolationInfos.push(groupInterpolationInfo_ContainsNumber);
                }
                else if (endsWithNumberInt && lastMatchInt) {
                    let groupInterpolationInfo_ContainsNumber = {
                        type: 'containsInt',
                        startIndexNumber: lastMatchInt.index,
                        endIndexNumber: lastMatchInt.index + lastMatchInt[0].length,
                        numberString: lastMatchInt[0]
                    };
                    groupInterpolationInfos.push(groupInterpolationInfo_ContainsNumber);
                }
                else if (monthNames.length === 1 || monthNamesFull.length === 1) {
                    let info;
                    let monthIndex = -1;
                    let isFullName;
                    let isUpperCase = cellText[0] !== cellText[0].toLowerCase();
                    if (monthNamesFull.length === 1) {
                        monthIndex = monthFullNames.indexOf(cellText.toLowerCase());
                        isFullName = true;
                    }
                    else {
                        monthIndex = monthShortNames.indexOf(cellText.toLowerCase());
                        isFullName = false;
                    }
                    if (monthIndex !== -1) {
                        info = {
                            type: 'month',
                            monthString: cellText,
                            monthIndex,
                            isFullName,
                            isUpperCase,
                        };
                    }
                    else {
                        console.warn(`Could not find month index for interpolation, defaulting to copying`);
                        info = {
                            type: 'unknown'
                        };
                    }
                    groupInterpolationInfos.push(info);
                }
                else {
                    let groupInterpolationInfo_Unknown = {
                        type: 'unknown'
                    };
                    groupInterpolationInfos.push(groupInterpolationInfo_Unknown);
                }
            }
        }
    }
    const bigZero = Big(0);
    const interpolationNumberSequenceStrings = [];
    let currentNumberStringSequence = [];
    let dataIndexToInterpolationSequenceIndexNumbers = [];
    const interpolationSequenceModelsNumbers = [];
    const interpolationLastXValNumbers = [];
    {
        for (let _i = 0; _i < groupInterpolationInfos.length; _i++) {
            const el = groupInterpolationInfos[_i];
            if (el.type === `int`) {
                currentNumberStringSequence.push(_data[_i]);
                dataIndexToInterpolationSequenceIndexNumbers[_i] = interpolationNumberSequenceStrings.length;
            }
            else {
                dataIndexToInterpolationSequenceIndexNumbers[_i] = -1;
                if (currentNumberStringSequence.length > 0) {
                    interpolationNumberSequenceStrings.push(currentNumberStringSequence);
                    currentNumberStringSequence = [];
                }
            }
        }
        if (currentNumberStringSequence.length > 0) {
            interpolationNumberSequenceStrings.push(currentNumberStringSequence);
        }
        for (let i = 0; i < interpolationNumberSequenceStrings.length; i++) {
            const sequenceStrings = interpolationNumberSequenceStrings[i];
            let ints = sequenceStrings.map((p, index) => {
                let canonicalNumberString = getFirstCanonicalNumberStringInCell(p, numbersStyleToUse);
                if (canonicalNumberString === null) {
                    console.warn(`Could not get canonical number string for interpolation at selection index: ${index}, defaulting to 0`);
                    return bigZero;
                }
                let num;
                try {
                    num = Big(canonicalNumberString);
                }
                catch (error) {
                    console.warn(`Could not parse canonical number string for interpolation at selection index: ${index}, defaulting to 0`);
                    return bigZero;
                }
                return num;
            });
            let isSimpleIncrement = false;
            if (ints.length === 1) {
                if (isNormalDirection) {
                    ints.push(ints[0].add(1));
                }
                else {
                    ints.push(ints[0].sub(1));
                }
                isSimpleIncrement = true;
            }
            let dataPoints = ints.map((val, index) => [Big(index + 1), val]);
            let model = regression.linearBig(dataPoints, { precisionBig: 2 });
            interpolationSequenceModelsNumbers.push(model);
            if (isSimpleIncrement) {
                interpolationLastXValNumbers.push(dataPoints[dataPoints.length - 2][0]);
            }
            else {
                interpolationLastXValNumbers.push(dataPoints[dataPoints.length - 1][0]);
            }
        }
    }
    let interpolationIndices = Array.from({ length: interpolationNumberSequenceStrings.length }, (_, i) => {
        return interpolationLastXValNumbers[i];
    });
    const interpolationContainsNumberSequenceIndices = [];
    let currentSequenceContainsNumberIndices = [];
    let curentSequenceContainsNumberGroupIndices = [];
    let dataIndexToInterpolationSequenceIndexContainsNumber = [];
    let interpolationIndexToDataGroupIndexContainsNumber = [];
    const interpolationSequenceModelsContainsNumber = [];
    const containsNumberInterpolationIndices = [];
    {
        for (let _i = 0; _i < groupInterpolationInfos.length; _i++) {
            const el = groupInterpolationInfos[_i];
            if (el.type === `containsInt`) {
                currentSequenceContainsNumberIndices.push(el.numberString);
                dataIndexToInterpolationSequenceIndexContainsNumber[_i] = interpolationContainsNumberSequenceIndices.length;
                curentSequenceContainsNumberGroupIndices.push(_i);
            }
            else {
                dataIndexToInterpolationSequenceIndexContainsNumber[_i] = -1;
                if (currentSequenceContainsNumberIndices.length > 0) {
                    interpolationContainsNumberSequenceIndices.push(currentSequenceContainsNumberIndices);
                    interpolationIndexToDataGroupIndexContainsNumber.push(curentSequenceContainsNumberGroupIndices);
                    currentSequenceContainsNumberIndices = [];
                    curentSequenceContainsNumberGroupIndices = [];
                }
            }
        }
        if (currentSequenceContainsNumberIndices.length > 0) {
            interpolationContainsNumberSequenceIndices.push(currentSequenceContainsNumberIndices);
            interpolationIndexToDataGroupIndexContainsNumber.push(curentSequenceContainsNumberGroupIndices);
        }
        for (let i = 0; i < interpolationContainsNumberSequenceIndices.length; i++) {
            const sequenceStrings = interpolationContainsNumberSequenceIndices[i];
            let ints = sequenceStrings.map((p, index) => {
                let canonicalNumberString = getFirstCanonicalNumberStringInCell(p, numbersStyleToUse);
                if (canonicalNumberString === null) {
                    console.warn(`Could not get canonical number string for interpolation at selection index: ${index}, defaulting to 0`);
                    return bigZero;
                }
                let num;
                try {
                    num = Big(canonicalNumberString);
                }
                catch (error) {
                    console.warn(`Could not parse canonical number string for interpolation at selection index: ${index}, defaulting to 0`);
                    return bigZero;
                }
                return num;
            });
            let delta;
            if (ints.length === 1) {
                delta = isNormalDirection ? Big(1) : Big(-1);
            }
            else {
                delta = ints[1].sub(ints[0]);
            }
            let allDeltasAreTheSame = true;
            for (let j = 1; j < ints.length; j++) {
                let _delta = ints[j].sub(ints[j - 1]);
                if (_delta.eq(delta) === false) {
                    allDeltasAreTheSame = false;
                    break;
                }
            }
            if (!allDeltasAreTheSame) {
                const groupIndicesThatContributedToThisGroup = interpolationIndexToDataGroupIndexContainsNumber[i];
                for (let k = 0; k < groupIndicesThatContributedToThisGroup.length; k++) {
                    const groupIndex = groupIndicesThatContributedToThisGroup[k];
                    groupInterpolationInfos[groupIndex] = {
                        type: 'unknown',
                    };
                }
                continue;
            }
            interpolationSequenceModelsContainsNumber.push(delta);
            containsNumberInterpolationIndices.push(ints[ints.length - 1]);
        }
    }
    const interpolationMonthSequenceIndices = [];
    let currentSequenceMonthIndices = [];
    let curentSequenceMonthGroupIndices = [];
    let dataIndexToInterpolationSequenceIndexMonths = [];
    let interpolationIndexToDataGroupIndexMonths = [];
    const interpolationSequenceModelsMonths = [];
    const monthInterpolationIndices = [];
    {
        for (let _i = 0; _i < groupInterpolationInfos.length; _i++) {
            const el = groupInterpolationInfos[_i];
            if (el.type === `month`) {
                currentSequenceMonthIndices.push(el.monthIndex);
                curentSequenceMonthGroupIndices.push(_i);
                dataIndexToInterpolationSequenceIndexMonths[_i] = interpolationMonthSequenceIndices.length;
            }
            else {
                dataIndexToInterpolationSequenceIndexMonths[_i] = -1;
                if (currentSequenceMonthIndices.length > 0) {
                    interpolationMonthSequenceIndices.push(currentSequenceMonthIndices);
                    currentSequenceMonthIndices = [];
                    interpolationIndexToDataGroupIndexMonths.push(curentSequenceMonthGroupIndices);
                    curentSequenceMonthGroupIndices = [];
                }
            }
        }
        if (currentSequenceMonthIndices.length > 0) {
            interpolationMonthSequenceIndices.push(currentSequenceMonthIndices);
            interpolationIndexToDataGroupIndexMonths.push(curentSequenceMonthGroupIndices);
        }
        for (let i = 0; i < interpolationMonthSequenceIndices.length; i++) {
            const monthIndexSequence = interpolationMonthSequenceIndices[i];
            let delta;
            if (monthIndexSequence.length === 1) {
                delta = isNormalDirection ? 1 : -1;
            }
            else {
                delta = monthIndexSequence[1] - monthIndexSequence[0];
            }
            let allDeltasAreTheSame = true;
            for (let j = 1; j < monthIndexSequence.length; j++) {
                if (monthIndexSequence[j] - monthIndexSequence[j - 1] !== delta) {
                    allDeltasAreTheSame = false;
                    break;
                }
            }
            if (!allDeltasAreTheSame) {
                const groupIndicesThatContributedToThisGroup = interpolationIndexToDataGroupIndexMonths[i];
                for (let k = 0; k < groupIndicesThatContributedToThisGroup.length; k++) {
                    const groupIndex = groupIndicesThatContributedToThisGroup[k];
                    groupInterpolationInfos[groupIndex] = {
                        type: 'unknown',
                    };
                }
                continue;
            }
            interpolationSequenceModelsMonths.push(delta % 12);
            monthInterpolationIndices.push(monthIndexSequence[monthIndexSequence.length - 1]);
        }
    }
    const interpolationDatesData = [];
    let currentSequenceDatesData = [];
    let curentSequenceDateGroupData = [];
    let dataIndexToInterpolationSequenceIndexDates = [];
    let interpolationIndexToDataGroupIndexDates = [];
    const interpolationSequenceModelsDates = [];
    const dateInterpolationStart = [];
    const dateInterpolationCounts = [];
    {
        for (let _i = 0; _i < groupInterpolationInfos.length; _i++) {
            const el = groupInterpolationInfos[_i];
            if (el.type === `date`) {
                currentSequenceDatesData.push(el.originalDate);
                curentSequenceDateGroupData.push(_i);
                dataIndexToInterpolationSequenceIndexDates[_i] = interpolationDatesData.length;
            }
            else {
                dataIndexToInterpolationSequenceIndexDates[_i] = -1;
                if (currentSequenceDatesData.length > 0) {
                    interpolationDatesData.push(currentSequenceDatesData);
                    currentSequenceDatesData = [];
                    interpolationIndexToDataGroupIndexDates.push(curentSequenceDateGroupData);
                    curentSequenceDateGroupData = [];
                }
            }
        }
        if (currentSequenceDatesData.length > 0) {
            interpolationDatesData.push(currentSequenceDatesData);
            interpolationIndexToDataGroupIndexDates.push(curentSequenceDateGroupData);
        }
        for (let i = 0; i < interpolationDatesData.length; i++) {
            const interpolationDateGroup = interpolationDatesData[i];
            let deltas = [0, 0, 0];
            if (interpolationDateGroup.length === 1) {
                deltas = [isNormalDirection ? 1 : -1, 0, 0];
            }
            else {
                let prevEl = interpolationDateGroup[0];
                let el = interpolationDateGroup[1];
                let diffInDays = el.date() - prevEl.date();
                let diffInMonths = el.month() - prevEl.month();
                let diffInYears = el.year() - prevEl.year();
                if (diffInDays === 0 && diffInMonths === 0) {
                    deltas = [0, 0, diffInYears];
                }
                else if (diffInDays === 0) {
                    let deltaInMonths = el.diff(prevEl, 'month');
                    deltas = [0, deltaInMonths, 0];
                }
                else {
                    let deltaInDays = el.diff(prevEl, 'day');
                    deltas = [deltaInDays, 0, 0];
                }
            }
            let allDeltasAreTheSame = [true, true, true];
            for (let j = 2; j < interpolationDateGroup.length; j++) {
                let prevEl = interpolationDateGroup[j - 1];
                let el = interpolationDateGroup[j];
                let diffInDays = el.date() - prevEl.date();
                let diffInMonths = el.month() - prevEl.month();
                let diffInYears = el.year() - prevEl.year();
                if (deltas[0] === 0 && deltas[1] === 0 && diffInDays == 0 && diffInMonths === 0) {
                    if (diffInYears !== deltas[2]) {
                        allDeltasAreTheSame[2] = false;
                    }
                }
                else if (deltas[0] === 0 && diffInDays == 0) {
                    if (diffInYears !== deltas[2]) {
                        allDeltasAreTheSame[2] = false;
                    }
                    if (diffInMonths !== deltas[1]) {
                        allDeltasAreTheSame[1] = false;
                    }
                }
                else {
                    let deltaInDays = el.diff(prevEl, 'day');
                    if (deltaInDays !== deltas[0]) {
                        allDeltasAreTheSame[0] = false;
                    }
                }
            }
            if (allDeltasAreTheSame.some(p => p === false)) {
                const groupIndicesThatContributedToThisGroup = interpolationIndexToDataGroupIndexDates[i];
                for (let k = 0; k < groupIndicesThatContributedToThisGroup.length; k++) {
                    const groupIndex = groupIndicesThatContributedToThisGroup[k];
                    groupInterpolationInfos[groupIndex] = {
                        type: 'unknown',
                    };
                }
                continue;
            }
            interpolationSequenceModelsDates.push(deltas);
            dateInterpolationStart.push(interpolationDateGroup[interpolationDateGroup.length - 1]);
            dateInterpolationCounts.push(0);
        }
    }
    let interpolatedDataAsString = [];
    console.debug(`auto fill numbersStyleToUse`, numbersStyleToUse);
    console.debug(`auto fill groupInterpolationInfos`, groupInterpolationInfos);
    for (let i = 0; i < targetCount; i++) {
        let relativI = i % _data.length;
        let groupInterpolationInfo = groupInterpolationInfos[relativI];
        switch (groupInterpolationInfo.type) {
            case "int": {
                let sequenceIndex = dataIndexToInterpolationSequenceIndexNumbers[relativI];
                let model = interpolationSequenceModelsNumbers[sequenceIndex];
                interpolationIndices[sequenceIndex] = interpolationIndices[sequenceIndex].add(1);
                let nextXVal = interpolationIndices[sequenceIndex];
                let predictedVal = model.predict(nextXVal);
                let numString = formatBigJsNumber(predictedVal[1], numbersStyleToUse);
                interpolatedDataAsString.push(numString);
                break;
            }
            case "containsInt": {
                let sequenceIndex = dataIndexToInterpolationSequenceIndexContainsNumber[relativI];
                let delta = interpolationSequenceModelsContainsNumber[sequenceIndex];
                let currNumberToUse = containsNumberInterpolationIndices[sequenceIndex];
                let predictedVal = currNumberToUse.add(delta);
                let cellText = _data[relativI];
                let numString = formatBigJsNumber(predictedVal, numbersStyleToUse);
                let newCellText = cellText.substring(0, groupInterpolationInfo.startIndexNumber) + numString + cellText.substring(groupInterpolationInfo.endIndexNumber);
                interpolatedDataAsString.push(newCellText);
                containsNumberInterpolationIndices[sequenceIndex] = predictedVal;
                break;
            }
            case "month": {
                let sequenceIndex = dataIndexToInterpolationSequenceIndexMonths[relativI];
                let delta = interpolationSequenceModelsMonths[sequenceIndex];
                let currMonthIndex = monthInterpolationIndices[sequenceIndex];
                let isFullName = groupInterpolationInfo.isFullName;
                let isUpperCase = groupInterpolationInfo.isUpperCase;
                let newMonthIndex = currMonthIndex + delta;
                if (newMonthIndex < 0) {
                    newMonthIndex = 12 + newMonthIndex;
                }
                let nextMonthIndex = newMonthIndex % 12;
                let nextMonth = isFullName ? monthFullNames[nextMonthIndex] : monthShortNames[nextMonthIndex];
                if (isUpperCase) {
                    nextMonth = nextMonth[0].toUpperCase() + nextMonth.substring(1);
                }
                monthInterpolationIndices[sequenceIndex] = nextMonthIndex;
                interpolatedDataAsString.push(nextMonth);
                break;
            }
            case "date": {
                let sequenceIndex = dataIndexToInterpolationSequenceIndexDates[relativI];
                let delta = interpolationSequenceModelsDates[sequenceIndex];
                let currStartDate = dateInterpolationStart[sequenceIndex];
                let currCount = dateInterpolationCounts[sequenceIndex];
                let nextCount = currCount + 1;
                let nextDate = currStartDate
                    .add(delta[0] * nextCount, 'day')
                    .add(delta[1] * nextCount, 'month')
                    .add(delta[2] * nextCount, 'year');
                if (nextDate.isValid() === false) {
                    interpolatedDataAsString.push("INVALID DATE");
                }
                else {
                    dateInterpolationCounts[sequenceIndex] = nextCount;
                    let dateString = nextDate.format(groupInterpolationInfo.displayFormat);
                    interpolatedDataAsString.push(dateString);
                }
                break;
            }
            case "unknown": {
                interpolatedDataAsString.push(_data[relativI]);
                break;
            }
            default:
                notExhaustiveSwitch(groupInterpolationInfo);
        }
    }
    if (!isNormalDirection) {
        interpolatedDataAsString.reverse();
    }
    if (interpolatedDataAsString.some(p => typeof p !== 'string')) {
        return [];
    }
    console.debug(`auto fill data`, interpolatedDataAsString);
    return interpolatedDataAsString;
}
//# sourceMappingURL=autoFill.js.map