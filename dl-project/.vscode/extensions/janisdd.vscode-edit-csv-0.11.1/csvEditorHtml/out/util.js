"use strict";
function _getById(id) {
    const el = document.getElementById(id);
    if (!el) {
        _error(`could not find element with id '${id}'`);
        return null;
    }
    return el;
}
function ensuredSingleCharacterString(el) {
    if (el.value.length > 1) {
        el.value = el.value.substring(el.value.length - 1);
    }
}
function isCommentCell(value, csvReadConfig) {
    if (value === null)
        return false;
    if (typeof csvReadConfig.comments === 'string' && csvReadConfig.comments !== '') {
        return value.trimLeft().startsWith(csvReadConfig.comments);
    }
    return false;
}
function containAndExtractUrl(text) {
    urlRegex.lastIndex = 0;
    urlRegex.lastIndex = 0;
    let matches = urlRegex.exec(text);
    if (matches) {
        let urls = [];
        while (matches) {
            urls.push({
                url: matches[0],
                startIndex: matches.index,
                endIndex: matches.index + matches[0].length
            });
            matches = urlRegex.exec(text);
        }
        return urls;
    }
    return null;
}
function createCellValueWithUrlLinks(text, urls) {
    const aTags = urls.map(url => {
        const a = document.createElement('a');
        a.href = url.url;
        a.innerText = url.url;
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
        a.setAttribute(linkIsOpenableAttribute, '1');
        a.title = `${getOpenLinkModifierTooltopPart()} ${url.url}`;
        return a;
    });
    let htmlParts = [];
    let currIndex = 0;
    for (let i = 0; i < aTags.length; i++) {
        const aTag = aTags[i];
        const urlObj = urls[i];
        if (currIndex < urlObj.startIndex) {
            htmlParts.push(text.substring(currIndex, urlObj.startIndex));
        }
        aTag.addEventListener(`mouseover`, (e) => {
            aTag.classList.add('link-hovered');
            hoveredATag = aTag;
            if (isOpenLinkModifierPressed(e)) {
                aTag.classList.add(isOpenUrlKeyDownClass);
            }
        });
        aTag.addEventListener(`mouseout`, (e) => {
            aTag.classList.remove('link-hovered');
            aTag.classList.remove(isOpenUrlKeyDownClass);
            hoveredATag = null;
        });
        aTag.addEventListener(`click`, (e) => {
            if (isOpenLinkModifierPressed(e)) {
                return;
            }
            e.stopPropagation();
            e.preventDefault();
            return false;
        });
        htmlParts.push(aTag);
        currIndex = urlObj.endIndex;
    }
    if (currIndex < text.length) {
        htmlParts.push(text.substring(currIndex));
    }
    return htmlParts;
}
function _normalizeDataArray(csvParseResult, csvReadConfig, fillString = '') {
    const maxCols = csvParseResult.data.reduce((prev, curr) => curr.length > prev ? curr.length : prev, 0);
    let someRowWasExpanded = false;
    let firstRealRowExpandedWasFound = false;
    if (csvParseResult.cellIsQuotedInfo.length !== csvParseResult.data.length) {
        postVsWarning(`cell quote information length does not match data length, defaulting to setting 'newColumnQuoteInformationIsQuoted': ${newColumnQuoteInformationIsQuoted}`);
        csvParseResult.cellIsQuotedInfo = Array.from(Array(csvParseResult.data.length), () => Array(maxCols).fill(newColumnQuoteInformationIsQuoted));
    }
    for (let i = 0; i < csvParseResult.data.length; i++) {
        const row = csvParseResult.data[i];
        const quoteRowInfo = csvParseResult.cellIsQuotedInfo[i];
        if (isCommentCell(row[0], csvReadConfig) === false && firstRealRowExpandedWasFound === false) {
            firstRealRowExpandedWasFound = true;
            if (row.length < maxCols && csvParseResult.columnIsQuoted !== null) {
                csvParseResult.columnIsQuoted.push(...Array.from(Array(maxCols - row.length), (p, index) => newColumnQuoteInformationIsQuoted));
            }
        }
        if (row.length < maxCols) {
            row.push(...Array.from(Array(maxCols - row.length), (p, index) => fillString));
            if (row.length > 0 && isCommentCell(row[0], csvReadConfig) === false) {
                someRowWasExpanded = true;
            }
        }
        if (quoteRowInfo.length < maxCols) {
            quoteRowInfo.push(...Array.from(Array(maxCols - quoteRowInfo.length), (p, index) => newColumnQuoteInformationIsQuoted));
        }
    }
    if (someRowWasExpanded) {
        postSetEditorHasChanges(true);
    }
}
function _resolveInitiallyHiddenColumns(csvParseResult, csvReadConfig) {
    let firstRealRow = null;
    for (let i = 0; i < csvParseResult.data.length; i++) {
        const row = csvParseResult.data[i];
        if (isCommentCell(row[0], csvReadConfig) === false) {
            firstRealRow = row;
            break;
        }
    }
    initiallyHiddenColumnIndices = [];
    if (firstRealRow === null || !initialConfig)
        return;
    let initiallyHiddenColumnIndicesSet = new Set();
    let _initiallyHiddenColumnNames = initialConfig.initiallyHiddenColumnNames ?? [];
    let _initiallyHiddenColumnNumbers = initialConfig.initiallyHiddenColumnNumbers ?? [];
    if (_initiallyHiddenColumnNames.length === 0 && _initiallyHiddenColumnNumbers.length === 0)
        return;
    let colNameToIndicesMap = new Map();
    for (let i = 0; i < firstRealRow.length; i++) {
        const cell = firstRealRow[i];
        let indicesList = colNameToIndicesMap.get(cell);
        if (indicesList === undefined) {
            indicesList = [];
            colNameToIndicesMap.set(cell, indicesList);
        }
        indicesList.push(i);
    }
    for (let i = 0; i < _initiallyHiddenColumnNames.length; i++) {
        const colName = _initiallyHiddenColumnNames[i];
        if (!colNameToIndicesMap.has(colName))
            continue;
        let indicesToHide = colNameToIndicesMap.get(colName);
        if (indicesToHide) {
            indicesToHide.forEach(index => initiallyHiddenColumnIndicesSet.add(index));
        }
    }
    for (let i = 0; i < _initiallyHiddenColumnNumbers.length; i++) {
        let maybeColNumber = _initiallyHiddenColumnNumbers[i];
        if (typeof maybeColNumber === 'string') {
            let check = parseInt(maybeColNumber);
            if (isNaN(check))
                continue;
            maybeColNumber = check;
        }
        let colIndex = maybeColNumber - 1;
        if (colIndex >= 0) {
            initiallyHiddenColumnIndicesSet.add(colIndex);
        }
    }
    initiallyHiddenColumnIndices = Array.from(initiallyHiddenColumnIndicesSet);
    if (initiallyHiddenColumnIndices.length === firstRealRow.length) {
        console.warn(`initially hidden columns: all columns are hidden, this is not allowed`);
        initiallyHiddenColumnIndices = [];
    }
}
function _hideColumnByIndices(columnIndices) {
    if (!hot)
        return;
    for (let i = 0; i < columnIndices.length; i++) {
        const targetColIndex = columnIndices[i];
        const physicalColIndex = hot.toPhysicalColumn(targetColIndex);
        hiddenPhysicalColumnIndicesSorted.push(physicalColIndex);
        const manualColumnResizePlugin = hot.getPlugin('manualColumnResize');
        manualColumnResizePlugin.manualColumnWidths[physicalColIndex] = undefined;
    }
    hiddenPhysicalColumnIndicesSorted = hiddenPhysicalColumnIndicesSorted.sort();
    firstAndLastVisibleColumns = getFirstAndLastVisibleColumns();
    hot.render();
}
function _unhideAllColumns() {
    if (!hot)
        return;
    let manualColumnResizePlugin = hot.getPlugin('manualColumnResize');
    for (let i = 0; i < hiddenPhysicalColumnIndicesSorted.length; i++) {
        const visualColIndex = hot.toVisualColumn(hiddenPhysicalColumnIndicesSorted[i]);
        manualColumnResizePlugin.clearManualSize(visualColIndex);
    }
    hiddenPhysicalColumnIndicesSorted = [];
    firstAndLastVisibleColumns = getFirstAndLastVisibleColumns();
    hot.render();
}
function _getCommentIndices(data, csvReadConfig) {
    if (typeof csvReadConfig.comments !== "string")
        return [];
    let commentIndices = [];
    for (let i = 0; i < data.length; i++) {
        const row = data[i];
        if (row.length > 0 && row[0] !== null && isCommentCell(row[0], csvReadConfig)) {
            commentIndices.push(i);
        }
    }
    return commentIndices;
}
function getSpreadsheetColumnLabel(index) {
    return `column ${index + 1}`;
}
const COLUMN_LABEL_BASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const COLUMN_LABEL_BASE_LENGTH = COLUMN_LABEL_BASE.length;
function spreadsheetColumnLetterLabel(index) {
    let num = index;
    let columnLabel = '';
    while (num >= 0) {
        columnLabel = COLUMN_LABEL_BASE[num % 26] + columnLabel;
        num = Math.floor(num / 26) - 1;
    }
    return columnLabel;
}
function addColumn(selectNewColumn = true) {
    if (!hot)
        throw new Error('table was null');
    const numCols = hot.countCols();
    hot.alter('insert_col', numCols);
    checkIfHasHeaderReadOptionIsAvailable(false);
    const pos = hot.getSelected();
    if (pos && pos.length === 1) {
        if (selectNewColumn) {
            hot.selectCell(pos[0][0], numCols);
        }
    }
}
function addRow(selectNewRow = true) {
    if (!hot)
        throw new Error('table was null');
    const numRows = hot.countRows();
    hot.alter('insert_row', numRows);
    firstAndLastVisibleRows = getFirstAndLastVisibleRows();
    if (selectNewRow) {
        hot.selectCell(numRows, 0);
    }
    checkAutoApplyHasHeader();
}
function _getSelectedVisualRowIndex() {
    if (!hot)
        throw new Error('table was null');
    const selections = hot.getSelected();
    if (!selections?.length)
        return null;
    const firstSelection = selections[0];
    const rowIndex = firstSelection[0];
    return rowIndex;
}
function _getSelectedVisualColIndex() {
    if (!hot)
        throw new Error('table was null');
    const selections = hot.getSelected();
    if (!selections?.length)
        return null;
    const firstSelection = selections[0];
    const rowIndex = firstSelection[1];
    return rowIndex;
}
function insertRowAbove() {
    if (isReadonlyMode)
        return;
    _insertRowInternal(false);
}
function insertRowBelow() {
    if (isReadonlyMode)
        return;
    _insertRowInternal(true);
}
function _insertRowInternal(belowCurrRow) {
    if (!hot)
        throw new Error('table was null');
    const currRowIndex = _getSelectedVisualRowIndex();
    const currColIndex = _getSelectedVisualColIndex();
    if (currRowIndex === null || currColIndex === null)
        return;
    const targetRowIndex = currRowIndex + (belowCurrRow ? 1 : 0);
    hot.alter('insert_row', targetRowIndex);
    const focusBehavior = initialConfig?.insertRowBehavior ?? 'focusFirstCellNewRow';
    switch (focusBehavior) {
        case 'focusFirstCellNewRow': {
            hot.selectCell(targetRowIndex, 0);
            break;
        }
        case 'keepRowKeepColumn': {
            hot.selectCell(targetRowIndex + (belowCurrRow ? -1 : 1), currColIndex);
            break;
        }
        default: notExhaustiveSwitch(focusBehavior);
    }
    checkAutoApplyHasHeader();
}
function insertColLeft(selectNewCol = true, preserveSelectedRow = true) {
    if (isReadonlyMode)
        return;
    _insertColInternal(false);
}
function insertColRight(selectNewCol = true, preserveSelectedRow = true) {
    if (isReadonlyMode)
        return;
    _insertColInternal(true);
}
function _insertColInternal(afterCurrCol) {
    if (!hot)
        throw new Error('table was null');
    const currColIndex = _getSelectedVisualColIndex();
    const currRowIndex = _getSelectedVisualRowIndex();
    if (currRowIndex === null || currColIndex === null)
        return;
    const targetColIndex = currColIndex + (afterCurrCol ? 1 : 0);
    hot.alter('insert_col', targetColIndex);
    const focusBehavior = initialConfig?.insertColBehavior ?? 'keepRowKeepColumn';
    switch (focusBehavior) {
        case 'keepRowFocusNewColumn': {
            hot.selectCell(currRowIndex, targetColIndex);
            break;
        }
        case 'keepRowKeepColumn': {
            hot.selectCell(currRowIndex, targetColIndex + (afterCurrCol ? -1 : 1));
            break;
        }
        default: notExhaustiveSwitch(focusBehavior);
    }
}
function getIsCallRemoveRowContextMenuActionDisabled() {
    if (isReadonlyMode)
        return true;
    const selection = hot.getSelected();
    let allRowsAreSelected = false;
    if (selection) {
        const selectedRowsCount = Math.abs(selection[0][0] - selection[0][2]);
        allRowsAreSelected = hot.countRows() === selectedRowsCount + 1;
    }
    return hot.countRows() === 1 || allRowsAreSelected;
}
function pretendRemoveRowContextMenuActionClicked() {
    if (!hot)
        return;
    let isMenuActionDisabled = getIsCallRemoveRowContextMenuActionDisabled();
    if (isMenuActionDisabled)
        return;
    const activeEditor = hot.getActiveEditor();
    if (activeEditor && activeEditor.isOpened()) {
        activeEditor.finishEditing(true);
        activeEditor.close();
    }
    const currRowIndex = _getSelectedVisualRowIndex();
    if (currRowIndex === null)
        return;
    const selRanges = hot.getSelectedRange();
    const normalizedSelection = selRanges
        ? selRanges.map(range => ({
            start: range.getTopLeftCorner(),
            end: range.getBottomRightCorner(),
        }))
        : [];
    hot.getPlugin('contextMenu')?.executeCommand('remove_row', normalizedSelection);
}
function removeRow(index) {
    if (isReadonlyMode)
        return;
    if (!hot)
        throw new Error('table was null');
    hot.alter('remove_row', index);
    checkIfHasHeaderReadOptionIsAvailable(false);
}
function removeColumn(index) {
    if (isReadonlyMode)
        return;
    if (!hot)
        throw new Error('table was null');
    hot.alter('remove_col', index);
    checkIfHasHeaderReadOptionIsAvailable(false);
}
function commentValueAndUrlsRenderer(instance, td, row, col, prop, value, cellProperties) {
    let isCellWithUrls = initialConfig?.convertUrlsToLinkTags
        ? value && (value.indexOf("http://") >= 0 || value.indexOf("https://") >= 0)
        : false;
    if (isCellWithUrls) {
        const urls = containAndExtractUrl(value);
        if (urls !== null) {
            const htmlParts = createCellValueWithUrlLinks(value, urls);
            Handsontable.dom.empty(td);
            td.append(...htmlParts);
        }
        Handsontable.renderers.BaseRenderer.apply(this, arguments);
    }
    else {
        Handsontable.renderers.TextRenderer.apply(this, arguments);
    }
    if (highlightCsvComments) {
        if (value !== null && col === 0 && isCommentCell(value, defaultCsvReadOptions)) {
            if (td && td.nextSibling) {
                td.nextSibling.title = warningTooltipTextWhenCommentRowNotFirstCellIsUsed;
            }
            if (td && td.parentElement) {
                td.parentElement.classList.add('comment-row');
            }
        }
    }
    if (isCellWithUrls) {
        return td;
    }
}
Handsontable.renderers.registerRenderer('commentValueAndUrlsRenderer', commentValueAndUrlsRenderer);
function _setOption(targetOptions, options, optionName) {
    if (options.hasOwnProperty(optionName)) {
        if (targetOptions.hasOwnProperty(optionName) === false) {
            _error(`target options object has not property '${String(optionName)}'`);
            return;
        }
        targetOptions[optionName] = options[optionName];
    }
    else {
        _error(`options object has not property '${String(optionName)}'`);
    }
}
function setCsvReadOptionsInitial(options) {
    const keys = Object.keys(defaultCsvReadOptions);
    for (const key of keys) {
        _setOption(defaultCsvReadOptions, options, key);
    }
    const el1 = _getById('delimiter-string');
    el1.value = defaultCsvReadOptions.delimiter;
    const el3 = _getById('has-header');
    el3.checked = defaultCsvReadOptions._hasHeader;
    const el4 = _getById('comment-string');
    el4.value = defaultCsvReadOptions.comments === false ? '' : defaultCsvReadOptions.comments;
    const el5 = _getById('quote-char-string');
    el5.value = defaultCsvReadOptions.quoteChar;
    const el6 = _getById('escape-char-string');
    el6.value = defaultCsvReadOptions.escapeChar;
}
function setCsvWriteOptionsInitial(options) {
    const keys = Object.keys(defaultCsvWriteOptions);
    for (const key of keys) {
        _setOption(defaultCsvWriteOptions, options, key);
    }
    const el1 = _getById('has-header-write');
    el1.checked = defaultCsvWriteOptions.header;
    const el2 = _getById('delimiter-string-write');
    el2.value = defaultCsvWriteOptions.delimiter;
    const el3 = _getById('comment-string-write');
    el3.value = defaultCsvWriteOptions.comments === false ? '' : defaultCsvWriteOptions.comments;
    const el4 = _getById('quote-char-string-write');
    el4.value = defaultCsvWriteOptions.quoteChar;
    const el5 = _getById('escape-char-string-write');
    el5.value = defaultCsvWriteOptions.quoteChar;
    const el6 = _getById('quote-all-fields-write');
    el6.checked = defaultCsvWriteOptions.quoteAllFields;
}
function checkIfHasHeaderReadOptionIsAvailable(isInitialRender) {
    const data = getData();
    const el = hasHeaderReadOptionInput;
    let canSetOption = false;
    if (isInitialRender) {
        canSetOption = data.length > 1;
    }
    else {
        if (defaultCsvReadOptions._hasHeader) {
            canSetOption = data.length >= 1;
        }
        else {
            canSetOption = data.length > 1;
        }
    }
    if (canSetOption) {
        const firstRow = getFirstRowWithIndex();
        if (firstRow === null && !el.checked) {
            canSetOption = false;
        }
    }
    if (canSetOption) {
    }
    else {
        defaultCsvReadOptions._hasHeader = false;
        el.checked = false;
        return false;
    }
    return true;
}
function throttle(func, wait) {
    var context, args, result;
    var timeout = null;
    var previous = 0;
    var later = function () {
        previous = Date.now();
        timeout = null;
        result = func.apply(context, args);
        if (!timeout)
            context = args = null;
    };
    return function () {
        var now = Date.now();
        var remaining = wait - (now - previous);
        context = this;
        args = arguments;
        if (remaining <= 0 || remaining > wait) {
            if (timeout) {
                clearTimeout(timeout);
                timeout = null;
            }
            previous = now;
            result = func.apply(context, args);
            if (!timeout)
                context = args = null;
        }
        else if (!timeout) {
            timeout = setTimeout(later, remaining);
        }
        return result;
    };
}
function debounce(func, wait, immediate = false) {
    var timeout;
    return function () {
        var context = this, args = arguments;
        var later = function () {
            timeout = null;
            if (!immediate)
                func.apply(context, args);
        };
        var callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow)
            func.apply(context, args);
    };
}
function _error(text) {
    postVsError(text);
    throw new Error(text);
}
function setupAndApplyInitialConfigPart1(initialConfig, initialVars) {
    {
        _setIsWatchingSourceFileUiIndicator(initialVars.isWatchingSourceFile);
    }
    if (initialConfig === undefined) {
        toggleOptionsBar(true);
        showCommentsBtn.style.display = 'none';
        hideCommentsBtn.style.display = '';
        return;
    }
    highlightCsvComments = initialConfig.highlightCsvComments;
    enableWrapping = initialConfig.enableWrapping;
    initialColumnWidth = initialConfig.initialColumnWidth;
    newColumnQuoteInformationIsQuoted = initialConfig.newColumnQuoteInformationIsQuoted;
    fixedRowsTop = Math.max(initialConfig.initiallyFixedRowsTop, 0);
    fixedColumnsLeft = Math.max(initialConfig.initiallyFixedColumnsLeft, 0);
    disableBorders = initialConfig.disableBorders;
    if (disableBorders) {
        const style = document.createElement('style');
        style.type = 'text/css';
        style.innerHTML = `.vscode-dark td, th { border: 0px !important; }`;
        document.getElementsByTagName('head')[0].appendChild(style);
    }
    changeFontSizeInPx(initialConfig.fontSizeInPx);
    const copyReadOptions = {
        ...defaultCsvReadOptions
    };
    let _readOption_hasHeader = initialConfig.readOption_hasHeader === 'true' ? true : false;
    if (_readOption_hasHeader) {
        isFirstHasHeaderChangedEvent = true;
    }
    else {
        isFirstHasHeaderChangedEvent = false;
    }
    setCsvReadOptionsInitial({
        ...copyReadOptions,
        delimiter: initialConfig.readOption_delimiter,
        delimitersToGuess: initialConfig.readOption_delimitersToGuess,
        comments: initialConfig.readOption_comment,
        _hasHeader: _readOption_hasHeader,
        escapeChar: initialConfig.readOption_escapeChar,
        quoteChar: initialConfig.readOption_quoteChar,
    });
    const copyWriteOptions = {
        ...defaultCsvReadOptions
    };
    setCsvWriteOptionsInitial({
        ...copyWriteOptions,
        comments: initialConfig.writeOption_comment,
        delimiter: initialConfig.writeOption_delimiter,
        header: initialConfig.writeOption_hasHeader === 'true' ? true : false,
        escapeChar: initialConfig.writeOption_escapeChar,
        quoteChar: initialConfig.writeOption_quoteChar,
        quoteAllFields: initialConfig.quoteAllFields,
        quoteEmptyOrNullFields: initialConfig.quoteEmptyOrNullFields === 'true' ? true : false,
    });
    switch (initialConfig.optionsBarAppearance) {
        case 'expanded': {
            toggleOptionsBar(false);
            break;
        }
        case 'collapsed': {
            toggleOptionsBar(true);
            break;
        }
        default: {
            _error(`unknown optionsBarAppearance: ${initialConfig.optionsBarAppearance}`);
            notExhaustiveSwitch(initialConfig.optionsBarAppearance);
            break;
        }
    }
    if (initialConfig.initiallyHideComments) {
        showCommentsBtn.style.display = '';
        hideCommentsBtn.style.display = 'none';
    }
    else {
        showCommentsBtn.style.display = 'none';
        hideCommentsBtn.style.display = '';
    }
    fixedRowsTopInfoSpan.innerText = fixedRowsTop + '';
    fixedColumnsTopInfoSpan.innerText = fixedColumnsLeft + '';
    isReadonlyMode = initialConfig.initiallyIsInReadonlyMode;
    _updateToggleReadonlyModeUi();
    setNumbersStyleUi(initialConfig.initialNumbersStyle);
    for (let i = 0; i < document.styleSheets.length; i++) {
        const styleSheet = document.styleSheets[i];
        if (!styleSheet.href?.toLowerCase().endsWith(`dark.css`) &&
            !styleSheet.href?.toLowerCase().endsWith(`light.css`))
            continue;
        for (let j = 0; j < styleSheet.cssRules.length; j++) {
            const cssRule = styleSheet.cssRules[j];
            if (cssRule instanceof CSSStyleRule) {
                const cssStyleRule = cssRule;
                if (cssStyleRule.selectorText === `body.vscode-light`) {
                    cssStyleRule.style.setProperty(cssFgColorVariableName, initialConfig.lightThemeTextColor);
                }
                if (cssStyleRule.selectorText === `body.vscode-dark`) {
                    cssStyleRule.style.setProperty(cssFgColorVariableName, initialConfig.darkThemeTextColor);
                }
            }
        }
    }
}
function _getVsState() {
    if (!vscode)
        return _createDefaultVsState();
    const state = vscode.getState();
    if (!state)
        return _createDefaultVsState();
    return state;
}
function _createDefaultVsState() {
    return {
        previewIsCollapsed: true,
        readOptionIsCollapsed: true,
        writeOptionIsCollapsed: true
    };
}
function _setReadOptionCollapsedVsState(isCollapsed) {
    if (vscode) {
    }
}
function _setWriteOptionCollapsedVsState(isCollapsed) {
    if (vscode) {
    }
}
function _setPreviewCollapsedVsState(isCollapsed) {
    if (vscode) {
    }
}
function customSearchMethod(query, value) {
    if (query === null || query === undefined || value === null || value === undefined)
        return false;
    if (query === '')
        return false;
    if (!findWidgetInstance.findOptionMatchCaseCache) {
        value = value.toLowerCase();
        query = query.toLowerCase();
    }
    if (findWidgetInstance.findOptionTrimCellCache) {
        value = value.trim();
    }
    if (findWidgetInstance.findOptionUseRegexCache) {
        if (findWidgetInstance.findWidgetCurrRegex === null) {
            throw new Error('should not happen...');
        }
        let result = findWidgetInstance.findWidgetCurrRegex.exec(value);
        if (findWidgetInstance.findOptionMatchWholeCellCache) {
            if (result !== null && result.length > 0) {
                return result[0] === value;
            }
        }
        return result !== null;
    }
    else {
        if (findWidgetInstance.findOptionMatchWholeCellCache) {
            return value === query;
        }
        return value.indexOf(query) !== -1;
    }
}
function afterHandsontableCreated(hot) {
    const afterSelectionHandler = (row, column, row2, column2) => {
        if (getIsSidePanelCollapsed()) {
        }
        else {
            calculateStats(row, column, row2, column2);
        }
    };
    hot.addHook('afterSelection', afterSelectionHandler);
    const afterRowOrColsCountChangeHandler = () => {
        statRowsCount.innerText = `${hot.countRows()}`;
        statColsCount.innerText = `${hot.countCols()}`;
    };
    hot.addHook('afterRemoveRow', afterRowOrColsCountChangeHandler);
    hot.addHook('afterCreateRow', afterRowOrColsCountChangeHandler);
    hot.addHook('afterCreateCol', afterRowOrColsCountChangeHandler);
    hot.addHook('afterRemoveCol', afterRowOrColsCountChangeHandler);
    statSelectedRows.innerText = `${0}`;
    statSelectedCols.innerText = `${0}`;
    statSelectedNotEmptyCells.innerText = `${0}`;
    statSumOfNumbers.innerText = `${0}`;
    statSelectedCellsCount.innerText = `${0}`;
    statRowsCount.innerText = `${hot.countRows()}`;
    statColsCount.innerText = `${hot.countCols()}`;
}
function recalculateStats() {
    const selectedRanges = hot.getSelected();
    if (!selectedRanges)
        return;
    const firstRange = selectedRanges[0];
    calculateStats(...firstRange);
}
function _calculateStats(row, column, row2, column2) {
    let numbersStyleToUse = getNumbersStyleFromUi();
    let rowsCount = Math.abs(row2 - row) + 1;
    let colsCount = Math.abs(column2 - column) + 1;
    statSelectedRows.innerText = `${rowsCount}`;
    statSelectedCols.innerText = `${colsCount}`;
    statSelectedCellsCount.innerText = `${rowsCount * colsCount}`;
    let notEmptyCount = 0;
    let numbersSum = Big(0);
    let containsInvalidNumbers = false;
    let minR = Math.min(row, row2);
    let maxR = Math.max(row, row2);
    for (let index = minR; index <= maxR; index++) {
        const data = hot.getDataAtRow(index);
        let minC = Math.min(column, column2);
        let maxC = Math.max(column, column2);
        for (let i = minC; i <= maxC; i++) {
            const el = data[i];
            if (el !== '' && el !== null) {
                notEmptyCount++;
                if (!containsInvalidNumbers) {
                    const firstCanonicalNumberStringInCell = getFirstCanonicalNumberStringInCell(el, numbersStyleToUse);
                    if (firstCanonicalNumberStringInCell === null)
                        continue;
                    try {
                        let _num = Big(firstCanonicalNumberStringInCell);
                        numbersSum = numbersSum.plus(_num);
                    }
                    catch (error) {
                        console.warn(`could not create or add number to statSumOfNumbers at row: ${index}, col: ${i}`, error);
                        containsInvalidNumbers = true;
                    }
                }
            }
        }
    }
    statSelectedNotEmptyCells.innerText = `${notEmptyCount}`;
    statSumOfNumbers.innerText = containsInvalidNumbers
        ? `Some invalid num`
        : `${formatBigJsNumber(numbersSum, numbersStyleToUse)}`;
}
const calculateStats = throttle(_calculateStats, 300);
function getFirstCanonicalNumberStringInCell(cellValue, numbersStyle) {
    let cellContent = cellValue;
    let thousandSeparatorsMatches;
    while (thousandSeparatorsMatches = numbersStyle.thousandSeparatorReplaceRegex.exec(cellValue)) {
        let replaceContent = thousandSeparatorsMatches[0].replace(numbersStyle.thousandSeparator, '');
        cellContent = cellContent.replace(thousandSeparatorsMatches[0], replaceContent);
    }
    let numberRegexRes = numbersStyle.regex.exec(cellContent);
    if (!numberRegexRes || numberRegexRes.length === 0)
        return null;
    return numberRegexRes[0].replace(/\,/gm, '.');
}
const knownNumberStylesMap = {
    "en": {
        key: 'en',
        regex: /-?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?/,
        regexStartToEnd: /^-?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/,
        thousandSeparator: /(\,| )/gm,
        thousandSeparatorReplaceRegex: /((\,| )\d{3})+/gm
    },
    "non-en": {
        key: 'non-en',
        regex: /-?(\d+(\,\d*)?|\,\d+)(e[+-]?\d+)?/,
        regexStartToEnd: /^-?(\d+(\,\d*)?|\,\d+)(e[+-]?\d+)?$/,
        thousandSeparator: /(\.| )/gm,
        thousandSeparatorReplaceRegex: /((\.| )\d{3})+/gm
    }
};
function setNumbersStyleUi(numbersStyleToUse) {
    numbersStyleEnRadio.checked = false;
    numbersStyleNonEnRadio.checked = false;
    numbersStyleEnRadio.removeAttribute('checked');
    numbersStyleEnRadio.removeAttribute('checked');
    switch (numbersStyleToUse) {
        case 'en': {
            numbersStyleEnRadio.checked = true;
            numbersStyleEnRadio.setAttribute('checked', 'true');
            break;
        }
        case 'non-en': {
            numbersStyleNonEnRadio.checked = true;
            numbersStyleNonEnRadio.setAttribute('checked', 'true');
            break;
        }
        default:
            notExhaustiveSwitch(numbersStyleToUse);
    }
}
function getNumbersStyleFromUi() {
    let knownNumberStylesMap = {
        "en": {
            key: 'en',
            regex: /-?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?/,
            regexStartToEnd: /^-?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/,
            thousandSeparator: /(\,| )/gm,
            thousandSeparatorReplaceRegex: /((\,| )\d{3})+/gm
        },
        "non-en": {
            key: 'non-en',
            regex: /-?(\d+(\,\d*)?|\,\d+)(e[+-]?\d+)?/,
            regexStartToEnd: /^-?(\d+(\,\d*)?|\,\d+)(e[+-]?\d+)?$/,
            thousandSeparator: /(\.| )/gm,
            thousandSeparatorReplaceRegex: /((\.| )\d{3})+/gm
        }
    };
    if (numbersStyleEnRadio.checked)
        return knownNumberStylesMap['en'];
    if (numbersStyleNonEnRadio.checked)
        return knownNumberStylesMap['non-en'];
    postVsWarning(`Got unknown numbers style from ui, defaulting to 'en'`);
    return knownNumberStylesMap['en'];
}
const b = new Big(1);
function formatBigJsNumber(bigJsNumber, numbersStyleToUse) {
    switch (numbersStyleToUse.key) {
        case 'en': {
            bigJsNumber.format = {
                decimalSeparator: '.',
                groupSeparator: '',
            };
            break;
        }
        case 'non-en': {
            bigJsNumber.format = {
                decimalSeparator: ',',
                groupSeparator: '',
            };
            break;
        }
        default:
            notExhaustiveSwitch(numbersStyleToUse.key);
    }
    return bigJsNumber.toFormat();
}
function calcHotCellToSelectFromCurosPos(openTableAndSelectCellAtCursorPos, sourceFileCursorLineIndex, sourceFileCursorColumnIndex, isCursorPosAfterLastColumn, csvParseResult, csvReadOptions) {
    if (openTableAndSelectCellAtCursorPos === 'never'
        || sourceFileCursorLineIndex === null
        || sourceFileCursorColumnIndex === null
        || !csvParseResult
        || !csvParseResult.outLineIndexToCsvLineIndexMapping
        || !csvParseResult.outColumnIndexToCsvColumnIndexMapping) {
        return {
            rowIndex: 0,
            colIndex: 0,
        };
    }
    let modifySelectedRowBecauseIfHeaderBy = 0;
    let csvRowToSelect = 0;
    let csvColToSelect = 0;
    try {
        let currentCsvRowStartTextFileLineIndex = 0;
        let currentCsvRowEndTextFileLineIndex = 0;
        let currentCursorColAsCsvCol = sourceFileCursorColumnIndex ?? 0;
        if (sourceFileCursorLineIndex < csvParseResult.outLineIndexToCsvLineIndexMapping.length) {
            csvRowToSelect = csvParseResult.outLineIndexToCsvLineIndexMapping[sourceFileCursorLineIndex];
            currentCsvRowStartTextFileLineIndex = csvParseResult.outLineIndexToCsvLineIndexMapping.indexOf(csvRowToSelect);
            currentCsvRowEndTextFileLineIndex = currentCsvRowStartTextFileLineIndex;
            if (currentCsvRowStartTextFileLineIndex === -1) {
                throw new Error(`Could not find text line for csv row ${csvRowToSelect}`);
            }
            {
                for (let i = currentCsvRowStartTextFileLineIndex; i < csvParseResult.outLineIndexToCsvLineIndexMapping.length; i++) {
                    let csvRowIndex = csvParseResult.outLineIndexToCsvLineIndexMapping[i];
                    if (csvRowIndex === csvRowToSelect) {
                        currentCsvRowEndTextFileLineIndex = i;
                        continue;
                    }
                    break;
                }
            }
            if (currentCsvRowStartTextFileLineIndex !== currentCsvRowEndTextFileLineIndex) {
                let allLines = csvParseResult.originalContent.split('\n');
                let startLineToSearch = currentCsvRowStartTextFileLineIndex;
                let maxTextLineToSearch = sourceFileCursorLineIndex - 1;
                currentCursorColAsCsvCol = 0;
                for (let i = startLineToSearch; i <= maxTextLineToSearch; i++) {
                    const textFileLine = allLines[i];
                    currentCursorColAsCsvCol += textFileLine.length + 1;
                }
                currentCursorColAsCsvCol += sourceFileCursorColumnIndex ?? 0;
            }
        }
        if (openTableAndSelectCellAtCursorPos === 'initialOnly_correctRowAndColumn') {
            let fieldEndIndices = csvParseResult.outColumnIndexToCsvColumnIndexMapping[csvRowToSelect];
            let findColByEndIndex = true;
            if (isCursorPosAfterLastColumn) {
                if (currentCsvRowStartTextFileLineIndex !== currentCsvRowEndTextFileLineIndex) {
                    if (sourceFileCursorLineIndex === currentCsvRowEndTextFileLineIndex) {
                        csvColToSelect = fieldEndIndices.length - 1;
                        findColByEndIndex = false;
                    }
                    else {
                        findColByEndIndex = true;
                    }
                }
                else {
                    csvColToSelect = fieldEndIndices.length - 1;
                    findColByEndIndex = false;
                }
            }
            else {
                findColByEndIndex = true;
            }
            if (findColByEndIndex) {
                for (let i = 0; i < fieldEndIndices.length; i++) {
                    let colEndIndex = fieldEndIndices[i];
                    if (currentCursorColAsCsvCol <= colEndIndex) {
                        csvColToSelect = i;
                        break;
                    }
                }
            }
        }
        if (csvReadOptions._hasHeader) {
            let csvHeaderRowIndex = 0;
            for (let i = 0; i < csvParseResult.data.length; i++) {
                const csvRow = csvParseResult.data[i];
                if (csvRow.length > 0 && isCommentCell(csvRow[0], csvReadOptions) === false) {
                    csvHeaderRowIndex = i;
                    break;
                }
            }
            if (csvRowToSelect === csvHeaderRowIndex) {
                csvRowToSelect++;
                modifySelectedRowBecauseIfHeaderBy = -1;
            }
            else if (csvRowToSelect > csvHeaderRowIndex) {
                modifySelectedRowBecauseIfHeaderBy = -1;
            }
            else {
                if (hiddenPhysicalRowIndicesSorted.length > 0) {
                    csvRowToSelect = getNextRowIfCommentsAreHidden(csvHeaderRowIndex + 1);
                    modifySelectedRowBecauseIfHeaderBy = -1;
                }
            }
        }
    }
    catch (e) {
        console.warn(`[edit csv] could not select line ${sourceFileCursorLineIndex ?? -1} or column ${sourceFileCursorColumnIndex ?? -1}`, e);
        return {
            rowIndex: 0,
            colIndex: 0
        };
    }
    let selectCellPos = {
        rowIndex: Math.max(0, csvRowToSelect),
        colIndex: Math.max(0, csvColToSelect)
    };
    selectCellPos.rowIndex = getNextRowIfCommentsAreHidden(selectCellPos.rowIndex);
    selectCellPos.rowIndex += modifySelectedRowBecauseIfHeaderBy;
    try {
        selectCellPos.rowIndex = Math.max(0, Math.min(csvParseResult.data.length - 1, selectCellPos.rowIndex));
        let dataRow = csvParseResult.data[selectCellPos.rowIndex];
        selectCellPos.colIndex = Math.max(0, Math.min(dataRow.length - 1, selectCellPos.colIndex));
    }
    catch (e) {
        selectCellPos.rowIndex = 0;
        selectCellPos.colIndex = 0;
    }
    return selectCellPos;
}
function scrollToSelectedCell(hot, cellToSelect) {
    let viewportTopRowToShow = 0;
    if (cellToSelect.rowIndex !== 0) {
        let autoRowSizePlugin = hot.getPlugin('autoRowSize');
        let firstVisibleRow = autoRowSizePlugin.getFirstVisibleRow();
        let lastVisibleRow = autoRowSizePlugin.getLastVisibleRow();
        let visibleRowCount = lastVisibleRow - firstVisibleRow + 1;
        viewportTopRowToShow = cellToSelect.rowIndex - Math.floor(visibleRowCount / 2);
        if (viewportTopRowToShow < 0) {
            viewportTopRowToShow = firstVisibleRow;
        }
        let maxRowCount = hot.countRows();
        viewportTopRowToShow = Math.max(0, Math.min(viewportTopRowToShow, maxRowCount - 1));
    }
    let viewportLeftColToShow = 0;
    if (cellToSelect.colIndex !== 0) {
        let autoColumnSizePlugin = hot.getPlugin('autoColumnSize');
        let firstVisibleCol = autoColumnSizePlugin.getFirstVisibleColumn();
        let lastVisibleCol = autoColumnSizePlugin.getLastVisibleColumn();
        let colWidths = autoColumnSizePlugin.widths;
        let viewportAbsoluteLeftOffset = 0;
        let viewportWidth = 0;
        let cumulativeAbsoluteOffsets = [];
        let selectedCellLeftOffsetMiddle = 0;
        for (let i = 0; i <= lastVisibleCol; i++) {
            if (i < firstVisibleCol) {
                viewportAbsoluteLeftOffset += colWidths[i];
            }
            else {
                viewportWidth += colWidths[i];
            }
            if (i < cellToSelect.colIndex) {
                selectedCellLeftOffsetMiddle += colWidths[i];
            }
            else if (i === cellToSelect.colIndex) {
                selectedCellLeftOffsetMiddle += Math.floor(colWidths[i] / 2);
            }
            if (i === 0) {
                cumulativeAbsoluteOffsets.push(colWidths[i]);
            }
            else {
                cumulativeAbsoluteOffsets.push(colWidths[i] + cumulativeAbsoluteOffsets[i - 1]);
            }
        }
        if (firstVisibleCol === 0 && selectedCellLeftOffsetMiddle <= viewportWidth / 2) {
            viewportLeftColToShow = 0;
        }
        else {
            let viewportAbsoluteMiddle = viewportAbsoluteLeftOffset + Math.floor(viewportWidth / 2);
            let viewportMiddleDiff = selectedCellLeftOffsetMiddle - viewportAbsoluteMiddle;
            let neededViewporLeftOffsetAbsolute = viewportAbsoluteLeftOffset + viewportMiddleDiff;
            for (let i = 0; i <= lastVisibleCol; i++) {
                let cumulativeOffset = cumulativeAbsoluteOffsets[i];
                if (neededViewporLeftOffsetAbsolute <= cumulativeOffset) {
                    viewportLeftColToShow = i;
                    break;
                }
            }
        }
    }
    hot.scrollViewportTo(viewportTopRowToShow, viewportLeftColToShow, false, false);
}
function getHotScrollPosition(hot) {
    return {
        top: hot.view.activeWt.wtOverlays.topOverlay.getScrollPosition(),
        left: hot.view.activeWt.wtOverlays.leftOverlay.getScrollPosition()
    };
}
function setHotScrollPosition(hot, viewportOffsetInPx) {
    ;
    hot.view.activeWt.wtOverlays.topOverlay.setScrollPosition(viewportOffsetInPx.top);
    hot.view.activeWt.wtOverlays.leftOverlay.setScrollPosition(viewportOffsetInPx.left);
}
function storeHotSelectedCellAndScrollPosition() {
    if (!hot)
        return;
    let hotSelection = hot.getSelected();
    if (hotSelection && hotSelection.length > 0) {
        previousSelectedCell = {
            rowIndex: hotSelection[0][0],
            colIndex: hotSelection[0][1]
        };
    }
    previousViewportOffsets = getHotScrollPosition(hot);
    let manualRowResizePlugin = hot.getPlugin('manualRowResize');
    previousManualRowHeights = manualRowResizePlugin.manualRowHeights;
}
function getNextRowIfCommentsAreHidden(visualRowIndex) {
    if (!hot)
        return visualRowIndex;
    if (hiddenPhysicalRowIndicesSorted.length === 0)
        return visualRowIndex;
    const lastPossibleRowIndex = hot.countRows() - 1;
    for (let i = visualRowIndex; i <= lastPossibleRowIndex; i++) {
        let physicalIndex = hot.toPhysicalRow(i);
        let isRowCommentAndHidden = hiddenPhysicalRowIndicesSorted.indexOf(physicalIndex) !== -1;
        if (isRowCommentAndHidden) {
            continue;
        }
        return i;
    }
    for (let i = visualRowIndex; i >= 0; i--) {
        let physicalIndex = hot.toPhysicalRow(i);
        let isRowCommentAndHidden = hiddenPhysicalRowIndicesSorted.indexOf(physicalIndex) !== -1;
        if (isRowCommentAndHidden) {
            continue;
        }
        return i;
    }
    return 0;
}
function getOpenLinkModifierTooltopPart() {
    if (isBrowser) {
        if (isMacOpenLinkModifierKey) {
            return `(cmd + click)`;
        }
        return `(ctrl + click)`;
    }
    if (isMacOpenLinkModifierKey) {
        return `(cmd + click)`;
    }
    return `(ctrl + click)`;
}
function isOpenLinkModifierPressed(e) {
    if (isBrowser) {
        if (isMacOpenLinkModifierKey && e.metaKey)
            return true;
        if (!isMacOpenLinkModifierKey && e.ctrlKey)
            return true;
    }
    if (isMacOpenLinkModifierKey && e.metaKey)
        return true;
    if (!isMacOpenLinkModifierKey && e.ctrlKey)
        return true;
    return false;
}
function getFirstAndLastVisibleColumns() {
    if (!hot)
        return { first: 0, last: 0 };
    let firstVisibleCol = 0;
    let lastVisibleCol = 0;
    const lastPossibleColIndex = hot.countCols() - 1;
    for (let index = 0; index <= lastPossibleColIndex; index++) {
        let physicalColIndex = hot.toPhysicalColumn(index);
        let isColHidden = hiddenPhysicalColumnIndicesSorted.indexOf(physicalColIndex) !== -1;
        if (!isColHidden) {
            firstVisibleCol = index;
            break;
        }
    }
    for (let index = lastPossibleColIndex; index >= 0; index--) {
        let physicalColIndex = hot.toPhysicalColumn(index);
        let isColHidden = hiddenPhysicalColumnIndicesSorted.indexOf(physicalColIndex) !== -1;
        if (!isColHidden) {
            lastVisibleCol = index;
            break;
        }
    }
    return { first: firstVisibleCol, last: lastVisibleCol };
}
function getFirstAndLastVisibleRows() {
    if (!hot)
        return { first: 0, last: 0 };
    let firstVisibleRow = 0;
    let lastVisibleRow = 0;
    const lastPossibleRowIndex = hot.countRows() - 1;
    for (let index = 0; index <= lastPossibleRowIndex; index++) {
        let physicalRowIndex = hot.toPhysicalRow(index);
        let isRowHidden = hiddenPhysicalRowIndicesSorted.indexOf(physicalRowIndex) !== -1;
        if (!isRowHidden) {
            firstVisibleRow = index;
            break;
        }
    }
    for (let index = lastPossibleRowIndex; index >= 0; index--) {
        let physicalRowIndex = hot.toPhysicalRow(index);
        let isRowHidden = hiddenPhysicalRowIndicesSorted.indexOf(physicalRowIndex) !== -1;
        if (!isRowHidden) {
            lastVisibleRow = index;
            break;
        }
    }
    return { first: firstVisibleRow, last: lastVisibleRow };
}
//# sourceMappingURL=util.js.map