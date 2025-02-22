"use strict";
function toggleOptionsBar(shouldCollapse) {
    const el = _getById('options-bar-icon');
    if (shouldCollapse === undefined) {
        if (el.classList.contains('fa-chevron-down')) {
            shouldCollapse = true;
        }
        else {
            shouldCollapse = false;
        }
    }
    document.documentElement.style
        .setProperty('--extension-options-bar-display', shouldCollapse ? `none` : `block`);
    if (vscode) {
        const lastState = _getVsState();
        vscode.setState({
            ...lastState,
            previewIsCollapsed: shouldCollapse
        });
    }
    if (shouldCollapse) {
        el.classList.remove('fa-chevron-down');
        el.classList.add('fa-chevron-right');
        onResizeGrid();
        _setPreviewCollapsedVsState(shouldCollapse);
        return;
    }
    el.classList.add('fa-chevron-down');
    el.classList.remove('fa-chevron-right');
    onResizeGrid();
    _setPreviewCollapsedVsState(shouldCollapse);
}
function _applyHasHeader(displayRenderInformation, fromUndo = false) {
    const el = hasHeaderReadOptionInput;
    const autoApplyHasHeader = shouldApplyHasHeaderAfterRowsAdded;
    setShouldAutpApplyHasHeader(false);
    const elWrite = _getById('has-header-write');
    let func = () => {
        if (!hot)
            throw new Error('table was null');
        if (el.checked || autoApplyHasHeader) {
            const dataWithIndex = getFirstRowWithIndex();
            if (dataWithIndex === null) {
                const el3 = _getById('has-header');
                el3.checked = false;
                headerRowWithIndex = null;
                return;
            }
            if (fromUndo)
                return;
            headerRowWithIndex = dataWithIndex;
            el.checked = true;
            _updateHandsontableSettings({
                fixedRowsTop: fixedRowsTop,
                fixedColumnsLeft: fixedColumnsLeft,
            }, false, false);
            let hasAnyChangesBefore = getHasAnyChangesUi();
            let headerCellIsQuotedInfoPhysicalIndices = cellIsQuotedInfoPhysicalIndices[headerRowWithIndex.physicalIndex];
            cellIsQuotedInfoPhysicalIndicesHeaderRow = headerCellIsQuotedInfoPhysicalIndices;
            hot.alter('remove_row', headerRowWithIndex.physicalIndex);
            elWrite.checked = true;
            defaultCsvWriteOptions.header = true;
            defaultCsvReadOptions._hasHeader = true;
            if (isFirstHasHeaderChangedEvent) {
                if (hasAnyChangesBefore === false) {
                    _setHasUnsavedChangesUiIndicator(false);
                }
                isFirstHasHeaderChangedEvent = false;
            }
            let undoPlugin = hot.undoRedo;
            undoPlugin.clear();
            hot.render();
            return;
        }
        if (fromUndo)
            return;
        if (headerRowWithIndex === null) {
            throw new Error('could not insert header row');
        }
        let hasAnyChangesBefore = getHasAnyChangesUi();
        hot.alter('insert_row', headerRowWithIndex.physicalIndex);
        const visualRow = hot.toVisualRow(headerRowWithIndex.physicalIndex);
        const visualCol = hot.toVisualColumn(0);
        hot.populateFromArray(visualRow, visualCol, [[...headerRowWithIndex.row]]);
        cellIsQuotedInfoPhysicalIndices[headerRowWithIndex.physicalIndex] = cellIsQuotedInfoPhysicalIndicesHeaderRow;
        cellIsQuotedInfoPhysicalIndicesHeaderRow = [];
        headerRowWithIndex = null;
        elWrite.checked = false;
        defaultCsvWriteOptions.header = false;
        defaultCsvReadOptions._hasHeader = false;
        _updateHandsontableSettings({
            fixedRowsTop: fixedRowsTop,
            fixedColumnsLeft: fixedColumnsLeft,
        }, false, false);
        if (isFirstHasHeaderChangedEvent) {
            if (hasAnyChangesBefore === false) {
                _setHasUnsavedChangesUiIndicator(false);
            }
            isFirstHasHeaderChangedEvent = false;
        }
        let undoPlugin = hot.undoRedo;
        undoPlugin.clear();
        hot.render();
    };
    if (displayRenderInformation) {
        statusInfo.innerText = `Rendering table...`;
        call_after_DOM_updated(() => {
            func();
            setTimeout(() => {
                statusInfo.innerText = '';
            }, 0);
        });
        return;
    }
    func();
}
function setShouldAutpApplyHasHeader(shouldSet) {
    if (shouldSet) {
        shouldApplyHasHeaderAfterRowsAdded = true;
        hasHeaderReadOptionInput.classList.add(`toggle-auto-future`);
        hasHeaderLabel.title = `Activated automatically, if table has >= 2 rows`;
    }
    else {
        hasHeaderReadOptionInput.classList.remove(`toggle-auto-future`);
        shouldApplyHasHeaderAfterRowsAdded = false;
        hasHeaderLabel.title = ``;
    }
}
function checkAutoApplyHasHeader() {
    if (!shouldApplyHasHeaderAfterRowsAdded)
        return;
    tryApplyHasHeader();
}
function tryApplyHasHeader() {
    if (!hot)
        return;
    const uiShouldApply = hasHeaderReadOptionInput.checked;
    const canApply = checkIfHasHeaderReadOptionIsAvailable(false);
    if (uiShouldApply) {
        if (!canApply) {
            if (shouldApplyHasHeaderAfterRowsAdded) {
                setShouldAutpApplyHasHeader(false);
                return;
            }
            setShouldAutpApplyHasHeader(true);
            return;
        }
    }
    _applyHasHeader(true, false);
}
function setDelimiterString() {
    const el = _getById('delimiter-string');
    defaultCsvReadOptions.delimiter = el.value;
}
function setCommentString() {
    const el = _getById('comment-string');
    defaultCsvReadOptions.comments = el.value === '' ? false : el.value;
}
function setQuoteCharString() {
    const el = _getById('quote-char-string');
    ensuredSingleCharacterString(el);
    defaultCsvReadOptions.quoteChar = el.value;
}
function setEscapeCharString() {
    const el = _getById('escape-char-string');
    ensuredSingleCharacterString(el);
    defaultCsvReadOptions.escapeChar = el.value;
}
function setSkipEmptyLines() {
}
function setReadDelimiter(delimiter) {
    const el = _getById('delimiter-string');
    el.value = delimiter;
    defaultCsvReadOptions.delimiter = delimiter;
}
function setHasHeaderWrite() {
    const el = _getById('has-header-write');
    defaultCsvWriteOptions.header = el.checked;
}
function setDelimiterStringWrite() {
    const el = _getById('delimiter-string-write');
    defaultCsvWriteOptions.delimiter = el.value;
}
function setCommentStringWrite() {
    const el = _getById('comment-string-write');
    defaultCsvWriteOptions.comments = el.value === '' ? false : el.value;
}
function setQuoteCharStringWrite() {
    const el = _getById('quote-char-string-write');
    ensuredSingleCharacterString(el);
    defaultCsvWriteOptions.quoteChar = el.value;
}
function setEscapeCharStringWrite() {
    const el = _getById('escape-char-string-write');
    ensuredSingleCharacterString(el);
    defaultCsvWriteOptions.escapeChar = el.value;
}
function setQuoteAllFieldsWrite() {
    const el = _getById('quote-all-fields-write');
    defaultCsvWriteOptions.quoteAllFields = el.checked;
}
function setNewLineWrite() {
    const el = _getById('newline-select-write');
    if (el.value === '') {
        defaultCsvWriteOptions.newline = newLineFromInput;
    }
    else if (el.value === 'lf') {
        defaultCsvWriteOptions.newline = '\n';
    }
    else if (el.value === 'crlf') {
        defaultCsvWriteOptions.newline = '\r\n';
    }
}
function setWriteDelimiter(delimiter) {
    const el = _getById('delimiter-string-write');
    el.value = delimiter;
    defaultCsvWriteOptions.delimiter = delimiter;
}
function generateCsvPreview() {
    const value = getDataAsCsv(defaultCsvReadOptions, defaultCsvWriteOptions);
    const el = _getById('csv-preview');
    el.value = value;
    toggleOptionsBar(false);
}
function copyPreviewToClipboard() {
    generateCsvPreview();
    const el = _getById('csv-preview');
    postCopyToClipboard(el.value);
}
function reRenderTable(callback) {
    if (!hot)
        return;
    statusInfo.innerText = `Rendering table...`;
    call_after_DOM_updated(() => {
        hot.render();
        setTimeout(() => {
            statusInfo.innerText = ``;
            if (callback) {
                setTimeout(() => {
                    callback();
                });
            }
        }, 0);
    });
}
function forceAutoResizeColumns() {
    if (!hot)
        return;
    let autoColumnSizePlugin = hot.getPlugin('autoColumnSize');
    let manualColumnResizePlugin = hot.getPlugin('manualColumnResize');
    let setColSizeFunc = () => {
        if (!hot)
            return;
        for (let i = 0; i < allColWidths.length; i++) {
            const colWidth = allColWidths[i];
            if (colWidth < 1)
                continue;
            manualColumnResizePlugin.clearManualSize(i);
        }
    };
    if (autoColumnSizePlugin.widths.length === 0) {
        autoColumnSizePlugin.enablePlugin();
    }
    setColSizeFunc();
    reRenderTable(setColSizeFunc);
}
function forceAutoResizeRows() {
    if (!hot)
        return;
    let plugin = hot.getPlugin('manualRowResize');
    plugin.manualRowHeights.forEach((height, physicalRowIndex) => {
        delete plugin.manualRowHeights[physicalRowIndex];
    });
    hot.forceFullRender = true;
    hot.view.render();
    hot.view.wt.wtOverlays.adjustElementsSize(true);
}
function displayData(csvParseResult, csvReadConfig) {
    if (csvParseResult === null) {
        if (hot) {
            hot.getInstance().destroy();
            hot = null;
        }
        return;
    }
    hiddenPhysicalColumnIndicesSorted = [];
    hiddenPhysicalRowIndicesSorted = [];
    firstAndLastVisibleColumns = null;
    firstAndLastVisibleRows = null;
    showOrHideAllComments(true);
    _normalizeDataArray(csvParseResult, csvReadConfig);
    columnIsQuoted = csvParseResult.columnIsQuoted;
    cellIsQuotedInfoPhysicalIndices = csvParseResult.cellIsQuotedInfo;
    _resolveInitiallyHiddenColumns(csvParseResult, csvReadConfig);
    headerRowWithIndex = null;
    const container = csvEditorDiv;
    if (hot) {
        hot.destroy();
        hot = null;
    }
    const initiallyHideComments = initialConfig ? initialConfig.initiallyHideComments : false;
    if (initiallyHideComments && typeof csvReadConfig.comments === 'string') {
        hiddenPhysicalRowIndicesSorted = _getCommentIndices(csvParseResult.data, csvReadConfig);
        hiddenPhysicalRowIndicesSorted = hiddenPhysicalRowIndicesSorted.sort();
    }
    findWidgetInstance.setupFind();
    const showColumnHeaderNamesWithLettersLikeExcel = initialConfig?.showColumnHeaderNamesWithLettersLikeExcel ?? false;
    let defaultColHeaderFuncBound = defaultColHeaderFunc.bind(this, showColumnHeaderNamesWithLettersLikeExcel);
    isInitialHotRender = true;
    hot = new Handsontable(container, {
        data: csvParseResult.data,
        readOnly: isReadonlyMode,
        trimWhitespace: false,
        rowHeaderWidth: getRowHeaderWidth(csvParseResult.data.length),
        renderAllRows: false,
        rowHeaders: function (row) {
            let text = (row + 1).toString();
            let showDeleteRowHeaderButton = initialConfig?.showDeleteRowHeaderButton ?? true;
            if (csvParseResult.data.length === 1 || isReadonlyMode || showDeleteRowHeaderButton === false) {
                return `${text} <span class="remove-row clickable" style="visibility: hidden"><i class="fas fa-trash"></i></span>`;
            }
            return `${text} <span class="remove-row clickable" onclick="removeRow(${row})"><i class="fas fa-trash"></i></span>`;
        },
        afterChange: onAnyChange,
        fillHandle: initialConfig?.dragToAutoFill !== 'none' ? {
            autoInsertRow: true,
        } : undefined,
        undo: true,
        colHeaders: defaultColHeaderFuncBound,
        currentColClassName: 'foo',
        currentRowClassName: 'foo',
        comments: false,
        autoWrapRow: initialConfig?.lastColumnOrFirstColumnNavigationBehavior === 'stop' ? false : true,
        autoWrapCol: initialConfig?.lastRowOrFirstRowNavigationBehavior === 'stop' ? false : true,
        search: {
            queryMethod: customSearchMethod,
            searchResultClass: 'search-result-cell',
        },
        wordWrap: enableWrapping,
        autoColumnSize: initialColumnWidth > 0 ? {
            maxColumnWidth: initialColumnWidth
        } : true,
        manualRowMove: true,
        manualRowResize: true,
        manualColumnMove: true,
        manualColumnResize: true,
        columnSorting: true,
        fixedRowsTop: fixedRowsTop,
        fixedColumnsLeft: fixedColumnsLeft,
        contextMenu: {
            items: {
                'row_above': {
                    callback: function () {
                        insertRowAbove();
                    },
                    disabled: function () {
                        return isReadonlyMode;
                    }
                },
                'row_below': {
                    callback: function () {
                        insertRowBelow();
                    },
                    disabled: function () {
                        return isReadonlyMode;
                    }
                },
                '---------': {
                    name: '---------'
                },
                'col_left': {
                    callback: function () {
                        insertColLeft();
                    },
                    disabled: function () {
                        return isReadonlyMode;
                    }
                },
                'col_right': {
                    callback: function () {
                        insertColRight();
                    },
                    disabled: function () {
                        return isReadonlyMode;
                    }
                },
                '---------2': {
                    name: '---------'
                },
                'remove_row': {
                    disabled: function () {
                        return getIsCallRemoveRowContextMenuActionDisabled();
                    },
                },
                'remove_col': {
                    disabled: function () {
                        if (isReadonlyMode)
                            return true;
                        const selection = hot.getSelected();
                        let allColsAreSelected = false;
                        if (selection) {
                            const selectedColsCount = Math.abs(selection[0][1] - selection[0][3]);
                            allColsAreSelected = hot.countCols() === selectedColsCount + 1;
                        }
                        return hot.countCols() === 1 || allColsAreSelected;
                    }
                },
                '---------3': {
                    name: '---------'
                },
                'alignment': {},
                'hide_columns': {
                    name: 'Hide column(s)',
                    callback: function (key, selection, clickEvent) {
                        if (!hot)
                            return;
                        if (selection.length > 1)
                            return;
                        let _selection = selection[0];
                        let columnIndicesToHide = [];
                        for (let targetCol = _selection.start.col; targetCol <= _selection.end.col; targetCol++) {
                            columnIndicesToHide.push(targetCol);
                        }
                        _hideColumnByIndices(columnIndicesToHide);
                    },
                    disabled: function () {
                        return isReadonlyMode;
                    }
                },
                'edit_header_cell': {
                    name: 'Edit header cell',
                    hidden: function () {
                        let selectedRanges = hot.getSelected();
                        if (selectedRanges?.length !== 1)
                            return true;
                        if (selectedRanges[0][1] !== selectedRanges[0][3])
                            return true;
                        let maxRowIndex = hot.countRows() - 1;
                        if (selectedRanges[0][0] !== 0 || selectedRanges[0][2] !== maxRowIndex)
                            return true;
                        if (!defaultCsvReadOptions._hasHeader)
                            return true;
                        if (!headerRowWithIndex)
                            return true;
                        return false;
                    },
                    callback: function (key, selection, clickEvent) {
                        if (!headerRowWithIndex)
                            return;
                        if (selection.length > 1)
                            return;
                        let targetCol = selection[0].start.col;
                        showColHeaderNameEditor(targetCol);
                    },
                    disabled: function () {
                        return isReadonlyMode;
                    }
                },
                'resize_row_header_cell': {
                    name: `Resize row to ${initialConfig?.doubleClickRowHandleForcedHeight ?? 106}px`,
                    callback: function (key, selection, clickEvent) {
                        if (!hot)
                            return;
                        let plugin = hot.getPlugin('manualRowResize');
                        let desiredColWidth = initialConfig?.doubleClickRowHandleForcedHeight ?? 106;
                        for (let i = selection[0].start.row; i <= selection[0].end.row; i++) {
                            plugin.setManualSize(i, desiredColWidth);
                        }
                        hot.forceFullRender = true;
                        hot.view.render();
                        hot.view.wt.wtOverlays.adjustElementsSize(true);
                    }
                },
                'resize_column_header_cell': {
                    name: `Resize column to ${initialConfig?.doubleClickColumnHandleForcedWith ?? 200}px`,
                    callback: function (key, selection, clickEvent) {
                        let plugin = hot.getPlugin('manualColumnResize');
                        let desiredColWidth = initialConfig?.doubleClickColumnHandleForcedWith ?? 200;
                        for (let i = selection[0].start.col; i <= selection[0].end.col; i++) {
                            plugin.setManualSize(i, desiredColWidth);
                        }
                        hot.forceFullRender = true;
                        hot.view.render();
                        hot.view.wt.wtOverlays.adjustElementsSize(true);
                    }
                },
                'unhide_all_columns': {
                    name: 'Unhide all columns',
                    callback: function (key, selection, clickEvent) {
                        _unhideAllColumns();
                    },
                    disabled: function () {
                        return hiddenPhysicalColumnIndicesSorted.length === 0;
                    }
                },
            }
        },
        beforeColumnSort: function (currentSortConfig, destinationSortConfigs) {
            if (isReadonlyMode)
                return false;
            return;
        },
        afterOnCellMouseUp: function () {
            if (editHeaderCellTextInputEl) {
                setTimeout(() => {
                    editHeaderCellTextInputEl.focus();
                }, 0);
            }
        },
        afterOnCellMouseDown: function (event, coords, th) {
            if (coords.row !== -1)
                return;
            lastClickedHeaderCellTh = th;
        },
        outsideClickDeselects: false,
        cells: highlightCsvComments
            ? function (row, col) {
                var cellProperties = {};
                cellProperties.renderer = 'commentValueAndUrlsRenderer';
                return cellProperties;
            }
            : undefined,
        beforeRowResize: function (oldSize, newSize, isDoubleClick) {
            if (oldSize === newSize) {
                if (initialConfig) {
                    return initialConfig.doubleClickRowHandleForcedHeight;
                }
                else {
                    console.log(`initialConfig is falsy`);
                }
            }
        },
        beforeColumnResize: function (oldSize, newSize, isDoubleClick) {
            if (oldSize === newSize) {
                if (initialConfig) {
                    return initialConfig.doubleClickColumnHandleForcedWith;
                }
                else {
                    console.log(`initialConfig is falsy`);
                }
            }
        },
        afterColumnResize: function () {
        },
        afterPaste: function () {
        },
        enterMoves: function (event) {
            if (!hot)
                throw new Error('table was null');
            lastHandsonMoveWas = 'enter';
            const selection = hot.getSelected();
            const _default = {
                row: 1,
                col: 0
            };
            if (!initialConfig || initialConfig.lastRowEnterBehavior !== 'createRow')
                return _default;
            if (!selection || selection.length === 0)
                return _default;
            if (selection.length > 1)
                return _default;
            const selected = selection[0];
            const isLastRowSelected = selected[0] === firstAndLastVisibleRows?.last;
            if (selected[0] !== selected[2] || !isLastRowSelected)
                return _default;
            if (event.key.toLowerCase() === 'enter' && event.shiftKey === false) {
                addRow(false);
            }
            return _default;
        },
        tabMoves: function (event) {
            if (!hot)
                throw new Error('table was null');
            lastHandsonMoveWas = 'tab';
            const selection = hot.getSelected();
            const _default = {
                row: 0,
                col: 1
            };
            if (!initialConfig || initialConfig.lastColumnTabBehavior !== 'createColumn')
                return _default;
            if (!selection || selection.length === 0)
                return _default;
            if (selection.length > 1)
                return _default;
            const selected = selection[0];
            const isLastColSelected = selected[1] === firstAndLastVisibleColumns?.last;
            if (selected[1] !== selected[3] || !isLastColSelected)
                return _default;
            if (event.key.toLowerCase() === 'tab' && event.shiftKey === false) {
                addColumn(false);
            }
            return _default;
        },
        afterBeginEditing: function () {
            if (!initialConfig || !initialConfig.selectTextAfterBeginEditCell)
                return;
            const textarea = document.getElementsByClassName("handsontableInput");
            if (!textarea || textarea.length === 0 || textarea.length > 1)
                return;
            const el = textarea.item(0);
            if (!el)
                return;
            el.setSelectionRange(0, el.value.length);
        },
        beforeCopy: function (data, coords) {
        },
        beforeUndo: function (_action) {
            let __action = _action;
            if (__action.actionType === 'changeHeaderCell' && headerRowWithIndex) {
                let action = __action;
                let visualColIndex = action.change[1];
                let beforeValue = action.change[2];
                let undoPlugin = hot.undoRedo;
                let undoneStack = undoPlugin.undoneActions;
                undoneStack.push(action);
                headerRowWithIndex.row[visualColIndex] = beforeValue;
                setTimeout(() => {
                    hot.render();
                }, 0);
                return false;
            }
            else if (__action.actionType === 'remove_col' && headerRowWithIndex) {
                let lastAction = headerRowWithIndexUndoStack.pop();
                if (lastAction && lastAction.action === "removed") {
                    headerRowWithIndex.row.splice(lastAction.visualIndex, 0, ...lastAction.headerData);
                    headerRowWithIndexRedoStack.push({
                        action: 'removed',
                        visualIndex: lastAction.visualIndex,
                        headerData: lastAction.headerData
                    });
                }
            }
            else if (__action.actionType === 'insert_col' && headerRowWithIndex) {
                let lastAction = headerRowWithIndexUndoStack.pop();
                if (lastAction && lastAction.action === "added") {
                    headerRowWithIndex.row.splice(lastAction.visualIndex, lastAction.headerData.length);
                    headerRowWithIndexRedoStack.push({
                        action: 'added',
                        visualIndex: lastAction.visualIndex,
                        headerData: lastAction.headerData
                    });
                }
            }
        },
        afterUndo: function (action) {
        },
        beforeRedo: function (_action) {
            let __action = _action;
            if (__action.actionType === 'changeHeaderCell' && headerRowWithIndex) {
                let action = __action;
                let visualColIndex = action.change[1];
                let afterValue = action.change[3];
                let undoPlugin = hot.undoRedo;
                let doneStack = undoPlugin.doneActions;
                doneStack.push(action);
                headerRowWithIndex.row[visualColIndex] = afterValue;
                setTimeout(() => {
                    hot.render();
                }, 0);
                return false;
            }
            else if (__action.actionType === 'remove_col' && headerRowWithIndex) {
                let lastAction = headerRowWithIndexRedoStack.pop();
                if (lastAction && lastAction.action === "removed") {
                    headerRowWithIndex.row.splice(lastAction.visualIndex, lastAction.headerData.length);
                    headerRowWithIndexUndoStack.push({
                        action: 'removed',
                        visualIndex: lastAction.visualIndex,
                        headerData: lastAction.headerData
                    });
                }
            }
            else if (__action.actionType === 'insert_col' && headerRowWithIndex) {
                let lastAction = headerRowWithIndexRedoStack.pop();
                if (lastAction && lastAction.action === "added") {
                    headerRowWithIndex.row.splice(lastAction.visualIndex, 0, ...lastAction.headerData);
                    headerRowWithIndexUndoStack.push({
                        action: 'added',
                        visualIndex: lastAction.visualIndex,
                        headerData: lastAction.headerData
                    });
                }
            }
        },
        beforeColumnMove: function (startColVisualIndex, endColVisualIndex) {
        },
        afterColumnMove: function (startColVisualIndices, endColVisualIndex) {
            if (!hot)
                throw new Error('table was null');
            headerRowWithIndexUndoStack.splice(0);
            headerRowWithIndexRedoStack.splice(0);
            let undoPlugin = hot.undoRedo;
            undoPlugin.clear();
            if (headerRowWithIndex !== null) {
                let temp = headerRowWithIndex;
                const headerRowTexts = startColVisualIndices.map(p => temp.row[p]);
                let headerRowCopy = [];
                for (let i = 0; i <= headerRowWithIndex.row.length; i++) {
                    const colText = i < headerRowWithIndex.row.length ? headerRowWithIndex.row[i] : null;
                    let startIndex = startColVisualIndices.indexOf(i);
                    if (startIndex !== -1) {
                        continue;
                    }
                    if (i === endColVisualIndex) {
                        headerRowCopy.push(...headerRowTexts);
                    }
                    if (i >= headerRowWithIndex.row.length)
                        continue;
                    headerRowCopy.push(colText);
                }
                headerRowWithIndex.row = headerRowCopy;
            }
            if (columnIsQuoted) {
                const startQuoteInformation = startColVisualIndices.map(p => columnIsQuoted[p]);
                let quoteCopy = [];
                for (let i = 0; i <= columnIsQuoted.length; i++) {
                    const quoteInfo = i < columnIsQuoted.length ? columnIsQuoted[i] : false;
                    let startIndex = startColVisualIndices.indexOf(i);
                    if (startIndex !== -1) {
                        continue;
                    }
                    if (i === endColVisualIndex) {
                        quoteCopy.push(...startQuoteInformation);
                    }
                    if (i >= columnIsQuoted.length)
                        continue;
                    quoteCopy.push(quoteInfo);
                }
                columnIsQuoted = quoteCopy;
            }
            onAnyChange();
        },
        beforeRowMove: function (startRow, endRow) {
        },
        afterRowMove: function (startRow, endRow) {
            if (!hot)
                throw new Error('table was null');
            onAnyChange();
        },
        afterGetRowHeader: function (visualRowIndex, th) {
            const tr = th.parentNode;
            if (!tr || !hot)
                return;
            let physicalIndex = hot.toPhysicalRow(visualRowIndex);
            if (hiddenPhysicalRowIndicesSorted.indexOf(physicalIndex) === -1) {
                tr.classList.remove('hidden-row');
                if (tr.previousElementSibling) {
                    tr.previousElementSibling.classList.remove('hidden-row-previous-row');
                }
            }
            else {
                tr.classList.add('hidden-row');
                if (tr.previousElementSibling) {
                    tr.previousElementSibling.classList.add('hidden-row-previous-row');
                }
            }
        },
        afterGetColHeader: function (visualColumnIndex, _th) {
            const th = _th;
            if (!th || !hot)
                return;
            let physicalIndex = hot.toPhysicalColumn(visualColumnIndex);
            if (hiddenPhysicalColumnIndicesSorted.indexOf(physicalIndex) === -1) {
                th.classList.remove('hidden-col');
                if (th.previousElementSibling) {
                    th.previousElementSibling.classList.remove('hidden-col-previous-col');
                }
            }
            else {
                th.classList.add('hidden-col');
                if (th.previousElementSibling) {
                    th.previousElementSibling.classList.add('hidden-col-previous-col');
                }
            }
        },
        afterCreateCol: function (visualColIndex, amount, source) {
            if (!hot)
                return;
            if (headerRowWithIndex) {
                if (source !== `UndoRedo.undo` && source !== `UndoRedo.redo`) {
                    headerRowWithIndexUndoStack.push({
                        action: 'added',
                        visualIndex: visualColIndex,
                        headerData: [...Array(amount).fill(null)],
                    });
                    headerRowWithIndex.row.splice(visualColIndex, 0, ...Array(amount).fill(null));
                }
            }
            if (columnIsQuoted) {
                columnIsQuoted.splice(visualColIndex, 0, ...Array(amount).fill(newColumnQuoteInformationIsQuoted));
            }
            if (cellIsQuotedInfoPhysicalIndices && cellIsQuotedInfoPhysicalIndices.length > 0) {
                for (let i = 0; i < amount; i++) {
                    let visualIndex = visualColIndex + i;
                    const physicalIndex = hot.toPhysicalColumn(visualIndex);
                    for (let j = 0; j < cellIsQuotedInfoPhysicalIndices.length; j++) {
                        const row = cellIsQuotedInfoPhysicalIndices[j];
                        row.splice(physicalIndex, 0, newColumnQuoteInformationIsQuoted);
                    }
                }
            }
            firstAndLastVisibleColumns = getFirstAndLastVisibleColumns();
            onAnyChange();
        },
        beforeRemoveCol(visualColIndex, amount, logicalCols) {
            if (!hot)
                return;
            if (cellIsQuotedInfoPhysicalIndices && cellIsQuotedInfoPhysicalIndices.length > 0) {
                for (let i = 0; i < amount; i++) {
                    let visualIndex = visualColIndex + i;
                    const physicalIndex = hot.toPhysicalColumn(visualIndex);
                    for (let j = 0; j < cellIsQuotedInfoPhysicalIndices.length; j++) {
                        const row = cellIsQuotedInfoPhysicalIndices[j];
                        row.splice(physicalIndex, 1);
                    }
                }
            }
            let physicalIndex = hot.toPhysicalColumn(visualColIndex);
            for (let i = 0; i < hiddenPhysicalColumnIndicesSorted.length; i++) {
                const hiddenPhysicalRowIndex = hiddenPhysicalColumnIndicesSorted[i];
                if (hiddenPhysicalRowIndex >= physicalIndex) {
                    hiddenPhysicalColumnIndicesSorted[i] -= amount;
                }
            }
        },
        afterRemoveCol: function (visualColIndex, amount, someting, source) {
            if (!hot)
                return;
            firstAndLastVisibleColumns = getFirstAndLastVisibleColumns();
            let isFromUndoRedo = (source === `UndoRedo.undo` || source === `UndoRedo.redo`);
            if (headerRowWithIndex && !isFromUndoRedo) {
                headerRowWithIndexUndoStack.push({
                    action: 'removed',
                    visualIndex: visualColIndex,
                    headerData: [headerRowWithIndex.row[visualColIndex]]
                });
            }
            pre_afterRemoveCol(visualColIndex, amount, isFromUndoRedo);
        },
        afterCreateRow: function (visualRowIndex, amount) {
            pre_afterCreateRow(visualRowIndex, amount);
        },
        beforeRemoveRow: function (visualRowIndex, amount) {
            if (!hot)
                return;
            let physicalIndex = hot.toPhysicalRow(visualRowIndex);
            cellIsQuotedInfoPhysicalIndices.splice(physicalIndex, amount);
            for (let i = 0; i < hiddenPhysicalRowIndicesSorted.length; i++) {
                const hiddenPhysicalRowIndex = hiddenPhysicalRowIndicesSorted[i];
                if (hiddenPhysicalRowIndex >= physicalIndex) {
                    hiddenPhysicalRowIndicesSorted[i] -= amount;
                }
            }
        },
        afterRemoveRow: function (visualRowIndex, amount) {
            if (!hot)
                return;
            firstAndLastVisibleRows = getFirstAndLastVisibleRows();
            if (headerRowWithIndex) {
                const lastValidIndex = hot.countRows();
                if (headerRowWithIndex.physicalIndex > lastValidIndex) {
                    headerRowWithIndex.physicalIndex = lastValidIndex;
                }
            }
            onAnyChange();
        },
        beforeSetRangeStartOnly: function (coords) {
        },
        beforeSetRangeStart: function (nextCoords) {
            if (!hot)
                return;
            if (hiddenPhysicalRowIndicesSorted.length === 0 && hiddenPhysicalColumnIndicesSorted.length === 0)
                return;
            const lastPossibleRowIndex = hot.countRows() - 1;
            const lastPossibleColIndex = hot.countCols() - 1;
            const actualSelection = hot.getSelectedLast();
            let columnIndexModifier = 0;
            let rowIndexModifier = 0;
            const isFirstRowHidden = hiddenPhysicalRowIndicesSorted.indexOf(hot.toPhysicalRow(0)) !== -1;
            const isLastRowHidden = hiddenPhysicalRowIndicesSorted.indexOf(hot.toPhysicalRow(lastPossibleRowIndex)) !== -1;
            const isFirstColHidden = hiddenPhysicalColumnIndicesSorted.indexOf(hot.toPhysicalColumn(0)) !== -1;
            const isLastColHidden = hiddenPhysicalColumnIndicesSorted.indexOf(hot.toPhysicalColumn(lastPossibleColIndex)) !== -1;
            const wrapNavigationAfterFirstOrLastRow = initialConfig?.lastRowOrFirstRowNavigationBehavior === 'wrap' ? true : false;
            const wrapNavigationAfterFirstOrLastCol = initialConfig?.lastColumnOrFirstColumnNavigationBehavior === 'wrap' ? true : false;
            let wasColWrapped = false;
            let wasRowWrapped = false;
            let directionRow = 0;
            let directionCol = 0;
            if (actualSelection) {
                const actualPhysicalIndex = actualSelection[0];
                directionRow = actualPhysicalIndex < nextCoords.row
                    ? 1
                    : actualPhysicalIndex > nextCoords.row
                        ? -1
                        : 0;
                wasRowWrapped = Math.abs(actualPhysicalIndex - nextCoords.row) > 1;
                const actualPhysicalColIndex = actualSelection[1];
                directionCol = actualPhysicalColIndex < nextCoords.col
                    ? 1
                    : actualPhysicalColIndex > nextCoords.col
                        ? -1
                        : 0;
                wasColWrapped = Math.abs(actualPhysicalColIndex - nextCoords.col) > 1;
            }
            const initialNavPos = {
                row: nextCoords.row,
                col: nextCoords.col
            };
            const getNextRow = (visualRowIndex) => {
                let visualRow = visualRowIndex;
                let physicalIndex = hot.toPhysicalRow(visualRowIndex);
                if (visualRow > lastPossibleRowIndex) {
                    if (wrapNavigationAfterFirstOrLastRow) {
                        if (directionRow === 1 && isLastRowHidden) {
                            columnIndexModifier = 1;
                        }
                        else if (directionRow === -1 && isFirstRowHidden) {
                            columnIndexModifier = -1;
                        }
                        return getNextRow(0);
                    }
                    return initialNavPos.row - directionRow;
                }
                if (visualRow < 0) {
                    if (wrapNavigationAfterFirstOrLastRow) {
                        if (directionRow === 1 && isLastRowHidden) {
                            columnIndexModifier = 1;
                        }
                        else if (directionRow === -1 && isFirstRowHidden) {
                            columnIndexModifier = -1;
                        }
                        return getNextRow(lastPossibleRowIndex);
                    }
                    return initialNavPos.row - directionRow;
                }
                if (hiddenPhysicalRowIndicesSorted.indexOf(physicalIndex) !== -1) {
                    if (directionRow === 0) {
                        return visualRow;
                    }
                    return getNextRow(visualRow + directionRow);
                }
                return visualRow;
            };
            const getNextCol = (visualColIndex) => {
                let visualCol = visualColIndex;
                let physicalIndex = hot.toPhysicalColumn(visualColIndex);
                if (visualCol > lastPossibleColIndex) {
                    if (wrapNavigationAfterFirstOrLastCol) {
                        if (directionCol === 1 && isLastColHidden) {
                            rowIndexModifier = 1;
                        }
                        else if (directionCol === -1 && isFirstColHidden) {
                            rowIndexModifier = -1;
                        }
                        return getNextCol(0);
                    }
                    return initialNavPos.col - directionCol;
                }
                if (visualCol < 0) {
                    if (wrapNavigationAfterFirstOrLastCol) {
                        if (directionCol === 1 && isLastColHidden) {
                            rowIndexModifier = 1;
                        }
                        else if (directionCol === -1 && isFirstColHidden) {
                            rowIndexModifier = -1;
                        }
                        return getNextCol(lastPossibleColIndex);
                    }
                    return initialNavPos.col - directionCol;
                }
                if (hiddenPhysicalColumnIndicesSorted.indexOf(physicalIndex) !== -1) {
                    if (directionCol === 0) {
                        return visualCol;
                    }
                    return getNextCol(visualCol + directionCol);
                }
                return visualCol;
            };
            if (directionRow !== 0 && directionCol !== 0) {
                if (wasColWrapped) {
                    directionCol = -directionCol;
                }
                if (wasRowWrapped) {
                    directionRow = -directionRow;
                }
                nextCoords.row = getNextRow(nextCoords.row);
                nextCoords.col = getNextCol(nextCoords.col);
            }
            else if (directionRow != 0) {
                nextCoords.row = getNextRow(nextCoords.row);
                if (wrapNavigationAfterFirstOrLastCol === false &&
                    (directionRow === -1 && nextCoords.col === firstAndLastVisibleColumns?.first
                        || directionRow === 1 && nextCoords.col === firstAndLastVisibleColumns?.last)) {
                }
                else {
                    nextCoords.col += columnIndexModifier;
                    directionCol = columnIndexModifier;
                    nextCoords.col = getNextCol(nextCoords.col);
                }
            }
            else if (directionCol != 0) {
                nextCoords.col = getNextCol(nextCoords.col);
                if (wrapNavigationAfterFirstOrLastRow === false &&
                    (directionCol === -1 && nextCoords.row === firstAndLastVisibleRows?.first
                        || directionCol === 1 && nextCoords.row === firstAndLastVisibleRows?.last)) {
                }
                else {
                    nextCoords.row += rowIndexModifier;
                    directionRow = rowIndexModifier;
                    nextCoords.row = getNextRow(nextCoords.row);
                }
            }
            lastHandsonMoveWas = null;
        },
        beforeSetRangeEnd: function () {
        },
        rowHeights: function (visualRowIndex) {
            let defaultHeight = (16 + fontSizeAddModifier) + 7;
            defaultHeight = Math.max(23, defaultHeight);
            if (!hot)
                return defaultHeight;
            const actualPhysicalIndex = hot.toPhysicalRow(visualRowIndex);
            if (hiddenPhysicalRowIndicesSorted.includes(actualPhysicalIndex)) {
                return 0.000001;
            }
            return defaultHeight;
        },
        colWidths: function (visualColIndex) {
            if (!hot)
                return undefined;
            const actualPhysicalIndex = hot.toPhysicalColumn(visualColIndex);
            if (hiddenPhysicalColumnIndicesSorted.includes(actualPhysicalIndex)) {
                return 0.000001;
            }
            return undefined;
        },
        beforeKeyDown: function (event) {
            if (editHeaderCellTextInputEl) {
                event.stopImmediatePropagation();
                return;
            }
            if ((event.ctrlKey || event.metaKey) && event.key === 'a' && findWidgetInstance.isFindWidgetDisplayed()) {
                event.stopImmediatePropagation();
                findWidgetInstance.selectAllInputText();
                return;
            }
            if (event.ctrlKey && event.shiftKey && event.altKey && event.key === 'ArrowDown') {
                event.stopImmediatePropagation();
                insertRowBelow();
            }
            else if (event.ctrlKey && event.shiftKey && event.altKey && event.key === 'ArrowUp') {
                event.stopImmediatePropagation();
                insertRowAbove();
            }
            else if (event.ctrlKey && event.shiftKey && event.altKey && event.key === 'ArrowLeft') {
                event.stopImmediatePropagation();
                insertColLeft();
            }
            else if (event.ctrlKey && event.shiftKey && event.altKey && event.key === 'ArrowRight') {
                event.stopImmediatePropagation();
                insertColRight();
            }
            if ((!isMacOS && event.altKey &&
                (event.key === `1`
                    || event.key === `2`
                    || event.key === `3`
                    || event.key === `4`
                    || event.key === `5`
                    || event.key === `6`
                    || event.key === `7`
                    || event.key === `8`
                    || event.key === `9`))
                || (event.ctrlKey && event.key === `Tab`)
                || (event.ctrlKey && event.shiftKey && event.key === `Tab`)) {
                hot?.setListeningPaused(true);
                setTimeout(() => {
                    hot?.setListeningPaused(false);
                }, 0);
                return;
            }
        },
        beforeOnCellMouseDown: function (event, coords, td, _blockCalculations) {
            if (isOpenLinkModifierPressed(event) && event.target && event.target.tagName.toLowerCase() === 'a') {
                let attrValue = event.target.getAttribute(linkIsOpenableAttribute);
                if (attrValue && attrValue === '1') {
                    event.stopImmediatePropagation();
                }
            }
        }
    });
    firstAndLastVisibleRows = getFirstAndLastVisibleRows();
    firstAndLastVisibleColumns = getFirstAndLastVisibleColumns();
    {
        let autoColumnSizePlugin = hot.getPlugin('autoColumnSize');
        autoColumnSizePlugin.ignoreCellWidthFunc = (value) => {
            const ignoreCommentCellWidths = initialConfig?.autoColumnWidthsIgnoreComments ?? true;
            const commentsAreHidden = !getAreCommentsDisplayed();
            return (ignoreCommentCellWidths || commentsAreHidden) && isCommentCell(value, csvReadConfig);
        };
    }
    let lastRowIndex = csvParseResult.data.length - 1;
    if (previousManualRowHeights) {
        let manualRowResizePlugin = hot.getPlugin('manualRowResize');
        previousManualRowHeights.forEach((height, physicalRowIndex) => {
            if (physicalRowIndex > lastRowIndex)
                return;
            manualRowResizePlugin.manualRowHeights[physicalRowIndex] = height;
        });
        previousManualRowHeights = null;
    }
    Handsontable.dom.addEvent(window, 'resize', throttle(onResizeGrid, 200));
    if (typeof afterHandsontableCreated !== 'undefined')
        afterHandsontableCreated(hot);
    hot.addHook('afterRender', afterRenderForced);
    let copyPastePlugin = hot.getPlugin('copyPaste');
    copyPastePlugin.rowsLimit = copyPasteRowLimit;
    copyPastePlugin.columnsLimit = copyPasteColLimit;
    copyPastePlugin.pasteSeparatorMode = initialConfig?.pasteMode || 'normal';
    copyPastePlugin.pasteMode = initialConfig?.pasteBehavior || `overwrite`;
    copyPastePlugin.pasteScrollBehavior = initialConfig?.pasteScrollBehavior || `scrollToLastPastedCell`;
    const oldShouldApplyHeaderReadOption = defaultCsvReadOptions._hasHeader;
    const settingsApplied = checkIfHasHeaderReadOptionIsAvailable(true);
    if (oldShouldApplyHeaderReadOption === true) {
        if (settingsApplied === true) {
            _applyHasHeader(true, false);
            updateFixedRowsCols();
        }
        else {
            setShouldAutpApplyHasHeader(true);
        }
    }
    isInitialHotRender = false;
    if (allColWidths && allColWidths.length > 0) {
        applyColWidths(true);
    }
    onResizeGrid();
    afterHandsontableCreated(hot);
    setupScrollListeners();
    if (initiallyHiddenColumnIndices.length > 0) {
        _hideColumnByIndices(initiallyHiddenColumnIndices);
    }
    if (hot) {
        let autoFillPlugin = hot.getPlugin('autofill');
        if (initialConfig?.dragToAutoFill === `excelLike`) {
            autoFillPlugin.setFillFunction(customAutoFillFunc);
        }
        if (previousSelectedCell === null || previousViewportOffsets === null) {
            let cellToSelect = {
                rowIndex: 0,
                colIndex: 0,
            };
            if (csvParseResult) {
                cellToSelect = calcHotCellToSelectFromCurosPos(initialVars.openTableAndSelectCellAtCursorPos, initialVars.sourceFileCursorLineIndex, initialVars.sourceFileCursorColumnIndex, initialVars.isCursorPosAfterLastColumn, csvParseResult, csvReadConfig);
            }
            else {
                cellToSelect = {
                    rowIndex: 0,
                    colIndex: 0
                };
            }
            hot.selectCell(cellToSelect.rowIndex, cellToSelect.colIndex);
            scrollToSelectedCell(hot, cellToSelect);
        }
        else {
            hot.selectCell(previousSelectedCell.rowIndex, previousSelectedCell.colIndex);
            setHotScrollPosition(hot, previousViewportOffsets);
        }
    }
}
function onAnyChange(changes, reason) {
    if (changes === null && reason && reason.toLowerCase() === 'loaddata') {
        return;
    }
    if (reason && reason === 'edit' && changes && changes.length > 0) {
        const hasChanges = changes.some(p => p[2] !== p[3]);
        if (!hasChanges)
            return;
    }
    if (findWidgetInstance.findWidgetInputValueCache !== '') {
        findWidgetInstance.tableHasChangedAfterSearch = true;
        findWidgetInstance.showOrHideOutdatedSearchIndicator(true);
    }
    postSetEditorHasChanges(true);
}
function onResizeGrid() {
    if (!hot)
        return;
    const widthString = getComputedStyle(csvEditorWrapper).width;
    if (!widthString) {
        _error(`could not resize table, width string was null`);
        return;
    }
    const width = parseInt(widthString.substring(0, widthString.length - 2));
    const heightString = getComputedStyle(csvEditorWrapper).height;
    if (!heightString) {
        _error(`could not resize table, height string was null`);
        return;
    }
    const height = parseInt(heightString.substring(0, heightString.length - 2));
    _updateHandsontableSettings({
        width: width,
        height: height,
    }, false, false);
    syncColWidths();
}
function applyColWidths(overwriteAutoSizedColumnWidths) {
    if (!hot)
        return;
    hot.getSettings().manualColumnResize = false;
    let autoSizedWidths = _getColWidths();
    for (let i = allColWidths.length; i < autoSizedWidths.length; i++) {
        const colWidth = autoSizedWidths[i];
        allColWidths.push(colWidth);
    }
    let manualColumnResizePlugin = hot.getPlugin('manualColumnResize');
    for (let i = 0; i < allColWidths.length; i++) {
        const colWidth = allColWidths[i];
        if (colWidth === undefined) {
            manualColumnResizePlugin.clearManualSize(i);
        }
        else {
            if (overwriteAutoSizedColumnWidths) {
                manualColumnResizePlugin.setManualSize(i, colWidth);
            }
        }
    }
    hot.getSettings().manualColumnResize = true;
    _updateHandsontableSettings({}, false, false);
}
function syncColWidths() {
    allColWidths = _getColWidths();
}
function _getColWidths() {
    if (!hot)
        return [];
    return hot.getColHeader().map(function (header, index) {
        return hot.getColWidth(index);
    });
}
function defaultColHeaderFunc(useLettersAsColumnNames, colIndex, colName) {
    let text = useLettersAsColumnNames
        ? spreadsheetColumnLetterLabel(colIndex)
        : getSpreadsheetColumnLabel(colIndex);
    if (headerRowWithIndex !== null && colIndex < headerRowWithIndex.row.length) {
        let visualIndex = colIndex;
        if (hot) {
            visualIndex = hot.toVisualColumn(colIndex);
        }
        const data = headerRowWithIndex.row[visualIndex];
        if (data !== null) {
            text = data;
        }
    }
    if (colName !== undefined && colName !== null) {
        text = colName;
    }
    let visualIndex = colIndex;
    if (!hot)
        return ``;
    visualIndex = hot.toVisualColumn(colIndex);
    let showDeleteColumnHeaderButton = initialConfig?.showDeleteColumnHeaderButton ?? true;
    if (hot.countCols() === 1 || isReadonlyMode || showDeleteColumnHeaderButton === false) {
        return `${text} <span class="remove-col clickable" style="visibility: hidden"><i class="fas fa-trash"></i></span>`;
    }
    return `${text} <span class="remove-col clickable" onclick="removeColumn(${visualIndex})"><i class="fas fa-trash"></i></span>`;
}
function toggleHelpModal(isVisible) {
    if (isVisible) {
        helModalDiv.classList.add('is-active');
        return;
    }
    helModalDiv.classList.remove('is-active');
}
function toggleAskReadAgainModal(isVisible) {
    if (isVisible) {
        askReadAgainModalDiv.classList.add('is-active');
        return;
    }
    askReadAgainModalDiv.classList.remove('is-active');
}
function toggleAskReloadFileModalDiv(isVisible) {
    if (isVisible) {
        askReloadFileModalDiv.classList.add('is-active');
        return;
    }
    askReloadFileModalDiv.classList.remove('is-active');
}
function toggleSourceFileChangedModalDiv(isVisible) {
    if (isVisible) {
        sourceFileChangedDiv.classList.add('is-active');
        return;
    }
    sourceFileChangedDiv.classList.remove('is-active');
}
function resetData(content, csvReadOptions) {
    const _data = parseCsv(content, csvReadOptions);
    displayData(_data, csvReadOptions);
    onResizeGrid();
    toggleAskReadAgainModal(false);
    setTimeout(() => {
        _updateHandsontableSettings({}, false, false);
    }, 0);
}
function resetDataFromResetDialog() {
    toggleAskReadAgainModal(false);
    postSetEditorHasChanges(false);
    storeHotSelectedCellAndScrollPosition();
    startRenderData();
}
function preReloadFileFromDisk() {
    const hasAnyChanges = getHasAnyChangesUi();
    if (hasAnyChanges) {
        toggleAskReloadFileModalDiv(true);
        return;
    }
    reloadFileFromDisk();
}
function reloadFileFromDisk() {
    toggleAskReloadFileModalDiv(false);
    toggleSourceFileChangedModalDiv(false);
    _setHasUnsavedChangesUiIndicator(false);
    storeHotSelectedCellAndScrollPosition();
    postReloadFile();
}
function startReceiveCsvProgBar() {
    receivedCsvProgBar.value = 0;
    receivedCsvProgBarWrapper.style.display = "block";
}
function intermediateReceiveCsvProgBar() {
    receivedCsvProgBar.attributes.removeNamedItem('value');
}
function stopReceiveCsvProgBar() {
    receivedCsvProgBarWrapper.style.display = "none";
}
function postApplyContent(saveSourceFile) {
    if (isReadonlyMode)
        return;
    const csvContent = getDataAsCsv(defaultCsvReadOptions, defaultCsvWriteOptions);
    if (document.activeElement !== document.body)
        document.activeElement.blur();
    _postApplyContent(csvContent, saveSourceFile);
}
function getRowHeaderWidth(rows) {
    const parentPadding = 5 * 2;
    const widthMultiplyFactor = 10;
    const iconPadding = 4;
    const binIcon = 14;
    const hiddenRowIcon = 10;
    const len = rows.toString().length * widthMultiplyFactor + binIcon + iconPadding + parentPadding + hiddenRowIcon;
    return len;
}
function trimAllCells() {
    if (!hot)
        throw new Error('table was null');
    const numRows = hot.countRows();
    const numCols = hot.countCols();
    const allData = getData();
    let data = '';
    let hasAnyChanges = false;
    let changeSet = [];
    for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < numCols; col++) {
            data = allData[row][col];
            if (typeof data !== "string") {
                continue;
            }
            allData[row][col] = data.trim();
            if (allData[row][col] !== data) {
                hasAnyChanges = true;
                changeSet.push([row, col, allData[row][col]]);
            }
        }
    }
    if (headerRowWithIndex) {
        for (let col = 0; col < headerRowWithIndex.row.length; col++) {
            const data = headerRowWithIndex.row[col];
            if (typeof data !== "string") {
                continue;
            }
            headerRowWithIndex.row[col] = data.trim();
            if (headerRowWithIndex.row[col] !== data) {
                hasAnyChanges = true;
            }
        }
    }
    hot.setDataAtCell(changeSet);
    if (hasAnyChanges) {
        onAnyChange();
    }
}
function transposeColumsAndRows() {
    if (!hot)
        return;
    const allData = getData();
    let transpose = allData[0].map((col, i) => allData.map(row => row[i]));
    let transposeQuoteInfo = cellIsQuotedInfoPhysicalIndices[0].map((col, i) => cellIsQuotedInfoPhysicalIndices.map(row => row[i]));
    cellIsQuotedInfoPhysicalIndices = transposeQuoteInfo;
    statusInfo.innerText = `Swapping finished, rendering...`;
    setTimeout(() => {
        statusInfo.innerText = '';
        _updateHandsontableSettings({
            data: transpose
        }, false, false);
        onAnyChange();
    }, 0);
}
function showOrHideAllComments(show) {
    if (show) {
        showCommentsBtn.style.display = 'none';
        hideCommentsBtn.style.display = '';
        hiddenPhysicalRowIndicesSorted = [];
    }
    else {
        showCommentsBtn.style.display = '';
        hideCommentsBtn.style.display = 'none';
        if (hot) {
            hiddenPhysicalRowIndicesSorted = _getCommentIndices(getData(), defaultCsvReadOptions);
            hiddenPhysicalRowIndicesSorted = hiddenPhysicalRowIndicesSorted.map(p => hot.toPhysicalRow(p));
            hiddenPhysicalRowIndicesSorted = hiddenPhysicalRowIndicesSorted.sort();
        }
    }
    firstAndLastVisibleRows = getFirstAndLastVisibleRows();
    if (!hot)
        return;
    hot.render();
}
function getAreCommentsDisplayed() {
    return showCommentsBtn.style.display === 'none';
}
function _setHasUnsavedChangesUiIndicator(hasUnsavedChanges) {
    if (hasUnsavedChanges) {
        unsavedChangesIndicator.classList.remove('op-hidden');
    }
    else {
        unsavedChangesIndicator.classList.add('op-hidden');
    }
}
function getHasAnyChangesUi() {
    return unsavedChangesIndicator.classList.contains("op-hidden") === false;
}
function _setIsWatchingSourceFileUiIndicator(isWatching) {
    if (isWatching) {
        sourceFileUnwatchedIndicator.classList.add('op-hidden');
    }
    else {
        sourceFileUnwatchedIndicator.classList.remove('op-hidden');
    }
}
function changeFontSizeInPx(fontSizeInPx) {
    document.documentElement.style.setProperty('--extension-font-size', `${fontSizeInPx.toString()}px`);
    if (fontSizeInPx <= 0) {
        document.body.classList.remove('extension-settings-font-size');
        document.body.classList.add('vs-code-settings-font-size');
        let fontSizeInPx = document.documentElement.style.getPropertyValue('--vscode-editor-font-size');
        let fontSize = parseInt(fontSizeInPx.substring(0, fontSizeInPx.length - 2));
        currentFontSize = isNaN(fontSize) ? 16 : fontSize;
    }
    else {
        document.body.classList.add('extension-settings-font-size');
        document.body.classList.remove('vs-code-settings-font-size');
        currentFontSize = fontSizeInPx;
    }
    reRenderTable();
}
function updateFixedRowsCols() {
    if (!hot)
        return;
    _updateHandsontableSettings({
        fixedRowsTop: Math.max(fixedRowsTop, 0),
        fixedColumnsLeft: Math.max(fixedColumnsLeft, 0),
    }, false, false);
}
function incFixedRowsTop() {
    _changeFixedRowsTop(fixedRowsTop + 1);
}
function decFixedRowsTop() {
    _changeFixedRowsTop(fixedRowsTop - 1);
}
function _changeFixedRowsTop(newVal) {
    fixedRowsTop = Math.max(newVal, 0);
    fixedRowsTopInfoSpan.innerText = fixedRowsTop.toString();
    updateFixedRowsCols();
}
function _toggleFixedRowsText() {
    const isHidden = fixedRowsTopText.classList.contains('dis-hidden');
    if (isHidden) {
        fixedRowsTopText.classList.remove('dis-hidden');
    }
    else {
        fixedRowsTopText.classList.add('dis-hidden');
    }
}
function incFixedColsLeft() {
    _changeFixedColsLeft(fixedColumnsLeft + 1);
}
function decFixedColsLeft() {
    _changeFixedColsLeft(fixedColumnsLeft - 1);
}
function _changeFixedColsLeft(newVal) {
    fixedColumnsLeft = Math.max(newVal, 0);
    fixedColumnsTopInfoSpan.innerText = fixedColumnsLeft.toString();
    updateFixedRowsCols();
}
function _toggleFixedColumnsText() {
    const isHidden = fixedColumnsTopText.classList.contains('dis-hidden');
    if (isHidden) {
        fixedColumnsTopText.classList.remove('dis-hidden');
    }
    else {
        fixedColumnsTopText.classList.add('dis-hidden');
    }
}
const minSidebarWidthInPx = 150;
const collapseSidePanelThreshold = 60;
function setupSideBarResizeHandle() {
    let downX = null;
    let _style = window.getComputedStyle(sidePanel);
    let downWidth = minSidebarWidthInPx;
    sideBarResizeHandle.addEventListener(`mousedown`, (e) => {
        downX = e.clientX;
        _style = window.getComputedStyle(sidePanel);
        downWidth = parseInt(_style.width.substring(0, _style.width.length - 2));
        if (isNaN(downWidth))
            downWidth = minSidebarWidthInPx;
    });
    document.addEventListener(`mousemove`, throttle((e) => {
        if (downX === null)
            return;
        const delta = e.clientX - downX;
        sidePanel.style.width = `${Math.max(downWidth + delta, minSidebarWidthInPx)}px`;
        sidePanel.style.maxWidth = `${Math.max(downWidth + delta, minSidebarWidthInPx)}px`;
        if (vscode) {
            const isSidePanelCollapsed = getIsSidePanelCollapsed();
            if (e.clientX <= collapseSidePanelThreshold) {
                if (!isSidePanelCollapsed)
                    toggleSidePanel(true);
            }
            else {
                if (isSidePanelCollapsed)
                    toggleSidePanel(false);
            }
        }
        onResizeGrid();
    }, 200));
    document.addEventListener(`mouseup`, (e) => {
        downX = null;
    });
}
function setupDropdownHandlers() {
    document.querySelectorAll(`.btn-with-menu`).forEach(btn => {
        btn.addEventListener(`mouseup`, (e) => {
            e.stopPropagation();
        });
    });
    document.addEventListener(`mouseup`, (e) => {
        let wrappers = document.querySelectorAll(`.btn-with-menu-wrapper`);
        wrappers.forEach(wrapper => {
            setToolMenuIsOpen(false);
        });
    });
}
function getHandsontableOverlayScrollLeft() {
    const overlayWrapper = document.querySelector(`#csv-editor-wrapper .ht_master .wtHolder`);
    if (!overlayWrapper) {
        console.warn(`could not find handsontable overlay wrapper`);
        return null;
    }
    return overlayWrapper;
}
function setupScrollListeners() {
    let overlayWrapper = getHandsontableOverlayScrollLeft();
    if (_onTableScrollThrottled) {
        overlayWrapper.removeEventListener(`scroll`, _onTableScrollThrottled);
    }
    _onTableScrollThrottled = throttle(_onTableScroll, 100);
    overlayWrapper.addEventListener(`scroll`, _onTableScrollThrottled);
}
function _onTableScroll(e) {
    if (!editHeaderCellTextInputEl)
        return;
    let scrollLeft = e.target.scrollLeft;
    editHeaderCellTextInputEl.style.left = `${editHeaderCellTextInputLeftOffsetInPx - (scrollLeft - handsontableOverlayScrollLeft)}px`;
}
function getIsSidePanelCollapsed() {
    if (vscode) {
        return window.getComputedStyle(leftPanelToggleIconExpand).display === 'block';
    }
    return false;
}
function toggleSidePanel(shouldCollapse) {
    if (vscode && shouldCollapse === undefined) {
        const isSidePanelCollapsed = getIsSidePanelCollapsed();
        if (isSidePanelCollapsed) {
            shouldCollapse = false;
        }
        else {
            shouldCollapse = true;
        }
    }
    document.documentElement.style
        .setProperty('--extension-side-panel-display', shouldCollapse ? `none` : `flex`);
    document.documentElement.style
        .setProperty('--extension-side-panel-expand-icon-display', shouldCollapse ? `block` : `none`);
    document.documentElement.style
        .setProperty('--extension-side-panel-collapse-icon-display', shouldCollapse ? `none` : `block`);
    onResizeGrid();
    if (shouldCollapse) {
    }
    else {
        recalculateStats();
    }
}
let recordedHookActions;
let hook_list = [];
function afterRenderForced(isForced) {
    if (!isForced) {
        hook_list = [];
        recordedHookActions = [];
        return;
    }
    for (let i = 0; i < hook_list.length; i++) {
        const hookItem = hook_list[i];
        if (!recordedHookActions.includes(hookItem.actionName))
            continue;
        hook_list.splice(i, 1);
        hookItem.action();
    }
    hook_list = [];
    recordedHookActions = [];
    if (!isForced || isInitialHotRender)
        return;
    syncColWidths();
}
function pre_afterRemoveCol(visualColIndex, amount, isFromUndoRedo) {
    recordedHookActions.push("afterRemoveCol");
    hook_list.push({
        actionName: 'afterRemoveCol',
        action: afterRemoveCol.bind(this, visualColIndex, amount, isFromUndoRedo)
    });
}
function afterRemoveCol(visualColIndex, amount, isFromUndoRedo) {
    if (!hot)
        return;
    if (headerRowWithIndex && !isFromUndoRedo) {
        headerRowWithIndex.row.splice(visualColIndex, amount);
    }
    const manualColumnResizePlugin = hot.getPlugin('manualColumnResize');
    const sortConfigs = hot.getPlugin('columnSorting').getSortConfig();
    const sortedColumnIds = sortConfigs.map(p => hot.toPhysicalColumn(p.column));
    let removedColIds = [];
    for (let i = 0; i < amount; i++) {
        removedColIds.push(hot.toPhysicalColumn(visualColIndex + i));
    }
    if (sortedColumnIds.some(p => removedColIds.includes(p))) {
        hot.getPlugin('columnSorting').clearSort();
    }
    if (columnIsQuoted) {
        columnIsQuoted.splice(visualColIndex, amount);
    }
    allColWidths.splice(visualColIndex, 1);
    const physicalColIndex = hot.toPhysicalColumn(visualColIndex);
    manualColumnResizePlugin.manualColumnWidths.splice(physicalColIndex, 1);
    applyColWidths(false);
    onAnyChange();
}
function pre_afterCreateRow(visualRowIndex, amount) {
    recordedHookActions.push("afterCreateRow");
    hook_list.push({
        actionName: 'afterCreateRow',
        action: afterCreateRow.bind(this, visualRowIndex, amount)
    });
}
function afterCreateRow(visualRowIndex, amount) {
    if (hot) {
        let physicalRowIndex = hot.toPhysicalRow(visualRowIndex);
        let numCols = hot.countCols();
        for (let i = 0; i < amount; i++) {
            let newRowQuoteInformation = Array.from(Array(numCols), () => newColumnQuoteInformationIsQuoted);
            cellIsQuotedInfoPhysicalIndices.splice(physicalRowIndex, 0, newRowQuoteInformation);
        }
    }
    for (let i = 0; i < hiddenPhysicalRowIndicesSorted.length; i++) {
        const hiddenPhysicalRowIndex = hiddenPhysicalRowIndicesSorted[i];
        if (hiddenPhysicalRowIndex >= visualRowIndex) {
            hiddenPhysicalRowIndicesSorted[i] += amount;
        }
    }
    firstAndLastVisibleRows = getFirstAndLastVisibleRows();
    onAnyChange();
    checkAutoApplyHasHeader();
}
function showColHeaderNameEditor(visualColIndex) {
    if (!headerRowWithIndex)
        return;
    if (!lastClickedHeaderCellTh)
        return;
    let rect = lastClickedHeaderCellTh.getBoundingClientRect();
    let input = document.createElement(`input`);
    input.setAttribute(`type`, `text`);
    input.style.position = `absolute`;
    input.style.left = `${rect.left}px`;
    editHeaderCellTextInputLeftOffsetInPx = rect.left;
    input.style.top = `${rect.top}px`;
    input.style.width = `${rect.width}px`;
    input.style.height = `${rect.height}px`;
    input.style.zIndex = `1000`;
    input.value = headerRowWithIndex.row[visualColIndex] ?? '';
    editHeaderCellTextInputEl = input;
    let overlayWrapper = getHandsontableOverlayScrollLeft();
    handsontableOverlayScrollLeft = overlayWrapper.scrollLeft;
    let inputWasRemoved = false;
    const removeInput = () => {
        editHeaderCellTextInputEl = null;
        if (inputWasRemoved)
            return;
        inputWasRemoved = true;
        input.remove();
    };
    const beforeValue = input.value;
    let shouldApplyChanges = true;
    let applyChange = () => {
        shouldApplyChanges = false;
        if (headerRowWithIndex && beforeValue !== input.value) {
            headerRowWithIndex.row[visualColIndex] = input.value;
            let undoPlugin = hot.undoRedo;
            let doneStack = undoPlugin.doneActions;
            let editHeaderRow = {
                actionType: 'changeHeaderCell',
                change: [0, visualColIndex, beforeValue, input.value]
            };
            doneStack.push(editHeaderRow);
            hot.render();
        }
    };
    let addListeners = () => {
        input.addEventListener(`blur`, () => {
            if (shouldApplyChanges) {
                applyChange();
            }
            removeInput();
        });
        input.addEventListener(`keyup`, (e) => {
            if (e.key.toLocaleLowerCase() === `escape`) {
                shouldApplyChanges = false;
                removeInput();
            }
            if (e.key.toLocaleLowerCase() === `enter`) {
                shouldApplyChanges = true;
                applyChange();
                removeInput();
            }
        });
    };
    document.body.appendChild(input);
    setTimeout(() => {
        addListeners();
    });
}
function _updateToggleReadonlyModeUi() {
    if (isReadonlyMode) {
        isReadonlyModeToggleSpan.classList.add(`active`);
        isReadonlyModeToggleSpan.title = `Sets the table to edit mode`;
        const btnEditableUi = document.querySelectorAll(`.on-readonly-disable-btn`);
        for (let i = 0; i < btnEditableUi.length; i++) {
            btnEditableUi.item(i).setAttribute('disabled', 'true');
        }
        const divEditableUi = document.querySelectorAll(`.on-readonly-disable-div`);
        for (let i = 0; i < divEditableUi.length; i++) {
            divEditableUi.item(i).classList.add('div-readonly-disabled');
        }
    }
    else {
        isReadonlyModeToggleSpan.classList.remove(`active`);
        isReadonlyModeToggleSpan.title = `Sets the table to readonly mode`;
        const btnEditableUi = document.querySelectorAll(`.on-readonly-disable-btn`);
        for (let i = 0; i < btnEditableUi.length; i++) {
            btnEditableUi.item(i).removeAttribute('disabled');
        }
        const divEditableUi = document.querySelectorAll(`.on-readonly-disable-div`);
        for (let i = 0; i < divEditableUi.length; i++) {
            divEditableUi.item(i).classList.remove('div-readonly-disabled');
        }
    }
}
function toggleReadonlyMode() {
    if (!hot)
        return;
    isReadonlyMode = !isReadonlyMode;
    _updateHandsontableSettings({
        readOnly: isReadonlyMode,
        manualRowMove: !isReadonlyMode,
        manualColumnMove: !isReadonlyMode,
        undo: !isReadonlyMode
    }, false, false);
    _updateToggleReadonlyModeUi();
}
function _updateHandsontableSettings(settings, init, skipEnablingAutoColumnSizePlugin) {
    if (!hot)
        return;
    hot.updateSettings(settings, init);
    if (skipEnablingAutoColumnSizePlugin)
        return;
    hot.getPlugin('autoColumnSize').enablePlugin();
}
function toggleToolMenu() {
    const isMenuOpen = toolMenuWrapper.classList.contains(`is-menu-open`);
    setToolMenuIsOpen(!isMenuOpen);
}
function setToolMenuIsOpen(setOpen) {
    if (setOpen) {
        toolMenuWrapper.classList.add(`is-menu-open`);
        toolsMenuBtnIcon.classList.remove(`fa-chevron-down`);
        toolsMenuBtnIcon.classList.add(`fa-chevron-up`);
    }
    else {
        toolMenuWrapper.classList.remove(`is-menu-open`);
        toolsMenuBtnIcon.classList.add(`fa-chevron-down`);
        toolsMenuBtnIcon.classList.remove(`fa-chevron-up`);
    }
}
let debouncedTableReRender = debounce(() => {
    reRenderTable(() => {
        hot.view.wt.wtOverlays.adjustElementsSize(true);
        hot.view.wt.draw();
        hot.view.wt.draw();
        let currEditor = hot.getActiveEditor();
        if (currEditor.state !== 'STATE_VIRGIN') {
            currEditor.refreshDimensions();
        }
    });
}, 1000);
function incTableContentZoom() {
    _changeTableContentZoom(2);
}
function decTableContentZoom() {
    _changeTableContentZoom(-2);
}
function resetTableContentZoom() {
    __changeTableContentZoom(0);
}
function _changeTableContentZoom(amount) {
    let newScaler = fontSizeAddModifier;
    newScaler += amount;
    if (currentFontSize + newScaler < 12) {
        return;
    }
    else if (currentFontSize + newScaler > 100) {
        return;
    }
    __changeTableContentZoom(newScaler);
}
function __changeTableContentZoom(newScalerFinal) {
    fontSizeAddModifier = newScalerFinal;
    document.documentElement.style.setProperty('--hot-font-size-add-modifier', `${fontSizeAddModifier}px`);
    hot.view.wt.draw();
    let currEditor = hot.getActiveEditor();
    if (currEditor.state !== 'STATE_VIRGIN') {
        currEditor.refreshDimensions();
    }
    debouncedTableReRender();
}
//# sourceMappingURL=ui.js.map