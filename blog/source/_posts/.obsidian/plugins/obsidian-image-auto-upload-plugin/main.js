'use strict';

var obsidian = require('obsidian');
var path = require('path');
var fs = require('fs');
var electron = require('electron');
var child_process = require('child_process');

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    if (typeof b !== "function" && b !== null)
        throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

function __values(o) {
    var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
    if (m) return m.call(o);
    if (o && typeof o.length === "number") return {
        next: function () {
            if (o && i >= o.length) o = void 0;
            return { value: o && o[i++], done: !o };
        }
    };
    throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
}

function __read(o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
}

function __asyncValues(o) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var m = o[Symbol.asyncIterator], i;
    return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i);
    function verb(n) { i[n] = o[n] && function (v) { return new Promise(function (resolve, reject) { v = o[n](v), settle(resolve, reject, v.done, v.value); }); }; }
    function settle(resolve, reject, d, v) { Promise.resolve(v).then(function(v) { resolve({ value: v, done: d }); }, reject); }
}

function isAnImage(ext) {
    return [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".svg", ".tiff"].includes(ext.toLowerCase());
}
function isAssetTypeAnImage(path$1) {
    return ([".png", ".jpg", ".jpeg", ".bmp", ".gif", ".svg", ".tiff"].indexOf(path.extname(path$1).toLowerCase()) !== -1);
}
function getOS() {
    var appVersion = navigator.appVersion;
    if (appVersion.indexOf("Win") !== -1) {
        return "Windows";
    }
    else if (appVersion.indexOf("Mac") !== -1) {
        return "MacOS";
    }
    else if (appVersion.indexOf("X11") !== -1) {
        return "Linux";
    }
    else {
        return "Unknown OS";
    }
}
function streamToString(stream) {
    var stream_1, stream_1_1;
    var e_1, _a;
    return __awaiter(this, void 0, void 0, function () {
        var chunks, chunk, e_1_1;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    chunks = [];
                    _b.label = 1;
                case 1:
                    _b.trys.push([1, 6, 7, 12]);
                    stream_1 = __asyncValues(stream);
                    _b.label = 2;
                case 2: return [4 /*yield*/, stream_1.next()];
                case 3:
                    if (!(stream_1_1 = _b.sent(), !stream_1_1.done)) return [3 /*break*/, 5];
                    chunk = stream_1_1.value;
                    chunks.push(Buffer.from(chunk));
                    _b.label = 4;
                case 4: return [3 /*break*/, 2];
                case 5: return [3 /*break*/, 12];
                case 6:
                    e_1_1 = _b.sent();
                    e_1 = { error: e_1_1 };
                    return [3 /*break*/, 12];
                case 7:
                    _b.trys.push([7, , 10, 11]);
                    if (!(stream_1_1 && !stream_1_1.done && (_a = stream_1.return))) return [3 /*break*/, 9];
                    return [4 /*yield*/, _a.call(stream_1)];
                case 8:
                    _b.sent();
                    _b.label = 9;
                case 9: return [3 /*break*/, 11];
                case 10:
                    if (e_1) throw e_1.error;
                    return [7 /*endfinally*/];
                case 11: return [7 /*endfinally*/];
                case 12: return [2 /*return*/, Buffer.concat(chunks).toString("utf-8")];
            }
        });
    });
}
function getUrlAsset(url) {
    return (url = url.substr(1 + url.lastIndexOf("/")).split("?")[0]).split("#")[0];
}
function isCopyImageFile() {
    var filePath = "";
    var os = getOS();
    if (os === "Windows") {
        var rawFilePath = electron.clipboard.read("FileNameW");
        filePath = rawFilePath.replace(new RegExp(String.fromCharCode(0), "g"), "");
    }
    else if (os === "MacOS") {
        filePath = electron.clipboard.read("public.file-url").replace("file://", "");
    }
    else {
        filePath = "";
    }
    return isAssetTypeAnImage(filePath);
}
function getLastImage(list) {
    var reversedList = list.reverse();
    var lastImage;
    reversedList.forEach(function (item) {
        if (item && item.startsWith("http")) {
            lastImage = item;
            return item;
        }
    });
    return lastImage;
}

var PicGoUploader = /** @class */ (function () {
    function PicGoUploader(settings) {
        this.settings = settings;
    }
    PicGoUploader.prototype.uploadFiles = function (fileList) {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, obsidian.requestUrl({
                            url: this.settings.uploadServer,
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ list: fileList }),
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.json;
                        return [2 /*return*/, data];
                }
            });
        });
    };
    PicGoUploader.prototype.uploadFileByClipboard = function () {
        return __awaiter(this, void 0, void 0, function () {
            var res, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, obsidian.requestUrl({
                            url: this.settings.uploadServer,
                            method: "POST",
                        })];
                    case 1:
                        res = _a.sent();
                        data = res.json;
                        if (res.status !== 200) {
                            ({ response: data, body: data.msg });
                            return [2 /*return*/, {
                                    code: -1,
                                    msg: data.msg,
                                    data: "",
                                }];
                        }
                        else {
                            return [2 /*return*/, {
                                    code: 0,
                                    msg: "success",
                                    data: data.result[0],
                                }];
                        }
                }
            });
        });
    };
    return PicGoUploader;
}());
var PicGoCoreUploader = /** @class */ (function () {
    function PicGoCoreUploader(settings) {
        this.settings = settings;
    }
    PicGoCoreUploader.prototype.uploadFiles = function (fileList) {
        return __awaiter(this, void 0, void 0, function () {
            var length, cli, command, res, splitList, splitListLength, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        length = fileList.length;
                        cli = this.settings.picgoCorePath || "picgo";
                        command = cli + " upload " + fileList
                            .map(function (item) { return "\"" + item + "\""; })
                            .join(" ");
                        return [4 /*yield*/, this.exec(command)];
                    case 1:
                        res = _a.sent();
                        splitList = res.split("\n");
                        splitListLength = splitList.length;
                        console.log(splitListLength);
                        data = splitList.splice(splitListLength - 1 - length, length);
                        if (res.includes("PicGo ERROR")) {
                            return [2 /*return*/, {
                                    success: false,
                                    msg: "失败",
                                }];
                        }
                        else {
                            return [2 /*return*/, {
                                    success: true,
                                    result: data,
                                }];
                        }
                }
            });
        });
    };
    // PicGo-Core 上传处理
    PicGoCoreUploader.prototype.uploadFileByClipboard = function () {
        return __awaiter(this, void 0, void 0, function () {
            var res, splitList, lastImage;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.uploadByClip()];
                    case 1:
                        res = _a.sent();
                        splitList = res.split("\n");
                        lastImage = getLastImage(splitList);
                        if (lastImage) {
                            return [2 /*return*/, {
                                    code: 0,
                                    msg: "success",
                                    data: lastImage,
                                }];
                        }
                        else {
                            new obsidian.Notice("\"Please check PicGo-Core config\"\n" + res);
                            return [2 /*return*/, {
                                    code: -1,
                                    msg: "\"Please check PicGo-Core config\"\n" + res,
                                    data: "",
                                }];
                        }
                }
            });
        });
    };
    // PicGo-Core的剪切上传反馈
    PicGoCoreUploader.prototype.uploadByClip = function () {
        return __awaiter(this, void 0, void 0, function () {
            var command, res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (this.settings.picgoCorePath) {
                            command = this.settings.picgoCorePath + " upload";
                        }
                        else {
                            command = "picgo upload";
                        }
                        return [4 /*yield*/, this.exec(command)];
                    case 1:
                        res = _a.sent();
                        return [2 /*return*/, res];
                }
            });
        });
    };
    PicGoCoreUploader.prototype.exec = function (command) {
        return __awaiter(this, void 0, void 0, function () {
            var stdout, res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, child_process.exec(command)];
                    case 1:
                        stdout = (_a.sent()).stdout;
                        return [4 /*yield*/, streamToString(stdout)];
                    case 2:
                        res = _a.sent();
                        return [2 /*return*/, res];
                }
            });
        });
    };
    return PicGoCoreUploader;
}());

var REGEX_FILE = /\!\[(.*?)\]\((.*?)\)/g;
var REGEX_WIKI_FILE = /\!\[\[(.*?)\]\]/g;
var Helper = /** @class */ (function () {
    function Helper(app) {
        this.app = app;
    }
    Helper.prototype.getFrontmatterValue = function (key, defaultValue) {
        if (defaultValue === void 0) { defaultValue = undefined; }
        var file = this.app.workspace.getActiveFile();
        if (!file) {
            return undefined;
        }
        var path = file.path;
        var cache = this.app.metadataCache.getCache(path);
        var value = defaultValue;
        if ((cache === null || cache === void 0 ? void 0 : cache.frontmatter) && cache.frontmatter.hasOwnProperty(key)) {
            value = cache.frontmatter[key];
        }
        return value;
    };
    Helper.prototype.getEditor = function () {
        var mdView = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
        if (mdView) {
            return mdView.editor;
        }
        else {
            return null;
        }
    };
    Helper.prototype.getValue = function () {
        var editor = this.getEditor();
        return editor.getValue();
    };
    Helper.prototype.setValue = function (value) {
        var editor = this.getEditor();
        var _a = editor.getScrollInfo(), left = _a.left, top = _a.top;
        var position = editor.getCursor();
        editor.setValue(value);
        editor.scrollTo(left, top);
        editor.setCursor(position);
    };
    // get all file urls, include local and internet
    Helper.prototype.getAllFiles = function () {
        var editor = this.getEditor();
        var value = editor.getValue();
        return this.getImageLink(value);
    };
    Helper.prototype.getImageLink = function (value) {
        var e_1, _a, e_2, _b;
        var matches = value.matchAll(REGEX_FILE);
        var WikiMatches = value.matchAll(REGEX_WIKI_FILE);
        var fileArray = [];
        try {
            for (var matches_1 = __values(matches), matches_1_1 = matches_1.next(); !matches_1_1.done; matches_1_1 = matches_1.next()) {
                var match = matches_1_1.value;
                var name_1 = match[1];
                var path$1 = match[2];
                var source = match[0];
                fileArray.push({
                    path: path$1,
                    name: name_1,
                    source: source,
                });
            }
        }
        catch (e_1_1) { e_1 = { error: e_1_1 }; }
        finally {
            try {
                if (matches_1_1 && !matches_1_1.done && (_a = matches_1.return)) _a.call(matches_1);
            }
            finally { if (e_1) throw e_1.error; }
        }
        try {
            for (var WikiMatches_1 = __values(WikiMatches), WikiMatches_1_1 = WikiMatches_1.next(); !WikiMatches_1_1.done; WikiMatches_1_1 = WikiMatches_1.next()) {
                var match = WikiMatches_1_1.value;
                console.log(match);
                var name_2 = path.parse(match[1]).name;
                var path$1 = match[1];
                var source = match[0];
                fileArray.push({
                    path: path$1,
                    name: name_2,
                    source: source,
                });
            }
        }
        catch (e_2_1) { e_2 = { error: e_2_1 }; }
        finally {
            try {
                if (WikiMatches_1_1 && !WikiMatches_1_1.done && (_b = WikiMatches_1.return)) _b.call(WikiMatches_1);
            }
            finally { if (e_2) throw e_2.error; }
        }
        return fileArray;
    };
    return Helper;
}());

var DEFAULT_SETTINGS = {
    uploadByClipSwitch: true,
    uploader: "PicGo",
    uploadServer: "http://127.0.0.1:36677/upload",
    picgoCorePath: "",
    menuMode: "auto",
    workOnNetWork: false,
};
var SettingTab = /** @class */ (function (_super) {
    __extends(SettingTab, _super);
    function SettingTab(app, plugin) {
        var _this = _super.call(this, app, plugin) || this;
        _this.plugin = plugin;
        return _this;
    }
    SettingTab.prototype.display = function () {
        var _this = this;
        var containerEl = this.containerEl;
        containerEl.empty();
        containerEl.createEl("h2", { text: "plugin settings" });
        new obsidian.Setting(containerEl)
            .setName("Auto pasted upload")
            .setDesc("if you set this value true, when you paste image, it will be auto uploaded(you should set the picGo server rightly)")
            .addToggle(function (toggle) {
            return toggle
                .setValue(_this.plugin.settings.uploadByClipSwitch)
                .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.plugin.settings.uploadByClipSwitch = value;
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        new obsidian.Setting(containerEl)
            .setName("Default uploader")
            .setDesc("Default uploader")
            .addDropdown(function (cb) {
            return cb
                .addOption("PicGo", "PicGo(app)")
                .addOption("PicGo-Core", "PicGo-Core")
                .setValue(_this.plugin.settings.uploader)
                .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.plugin.settings.uploader = value;
                            this.display();
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        if (this.plugin.settings.uploader === "PicGo") {
            new obsidian.Setting(containerEl)
                .setName("PicGo server")
                .setDesc("PicGo server")
                .addText(function (text) {
                return text
                    .setPlaceholder("Please input PicGo server")
                    .setValue(_this.plugin.settings.uploadServer)
                    .onChange(function (key) { return __awaiter(_this, void 0, void 0, function () {
                    return __generator(this, function (_a) {
                        switch (_a.label) {
                            case 0:
                                this.plugin.settings.uploadServer = key;
                                return [4 /*yield*/, this.plugin.saveSettings()];
                            case 1:
                                _a.sent();
                                return [2 /*return*/];
                        }
                    });
                }); });
            });
        }
        if (this.plugin.settings.uploader === "PicGo-Core") {
            new obsidian.Setting(containerEl)
                .setName("PicGo-Core path")
                .setDesc("Please input PicGo-Core path, default using environment variables")
                .addText(function (text) {
                return text
                    .setPlaceholder("")
                    .setValue(_this.plugin.settings.picgoCorePath)
                    .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                    return __generator(this, function (_a) {
                        switch (_a.label) {
                            case 0:
                                this.plugin.settings.picgoCorePath = value;
                                return [4 /*yield*/, this.plugin.saveSettings()];
                            case 1:
                                _a.sent();
                                return [2 /*return*/];
                        }
                    });
                }); });
            });
        }
        new obsidian.Setting(containerEl)
            .setName("Upload contextMenu mode")
            .setDesc("It should be set like your ob setting, otherwise the feature can not be work.")
            .addDropdown(function (cb) {
            return cb
                .addOption("auto", "auto(Read from config)")
                .addOption("absolute", "absolute")
                .addOption("relative", "relative")
                .setValue(_this.plugin.settings.menuMode)
                .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.plugin.settings.menuMode = value;
                            this.display();
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        new obsidian.Setting(containerEl)
            .setName("Work on network")
            .setDesc("When you paste, md standard image link in your clipboard will be auto upload.")
            .addToggle(function (toggle) {
            return toggle
                .setValue(_this.plugin.settings.workOnNetWork)
                .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.plugin.settings.workOnNetWork = value;
                            this.display();
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
    };
    return SettingTab;
}(obsidian.PluginSettingTab));

var imageAutoUploadPlugin = /** @class */ (function (_super) {
    __extends(imageAutoUploadPlugin, _super);
    function imageAutoUploadPlugin() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    imageAutoUploadPlugin.prototype.loadSettings = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, _b, _c, _d;
            return __generator(this, function (_e) {
                switch (_e.label) {
                    case 0:
                        _a = this;
                        _c = (_b = Object).assign;
                        _d = [DEFAULT_SETTINGS];
                        return [4 /*yield*/, this.loadData()];
                    case 1:
                        _a.settings = _c.apply(_b, _d.concat([_e.sent()]));
                        return [2 /*return*/];
                }
            });
        });
    };
    imageAutoUploadPlugin.prototype.saveSettings = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.saveData(this.settings)];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    imageAutoUploadPlugin.prototype.onunload = function () { };
    imageAutoUploadPlugin.prototype.onload = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.loadSettings()];
                    case 1:
                        _a.sent();
                        this.helper = new Helper(this.app);
                        this.picGoUploader = new PicGoUploader(this.settings);
                        this.picGoCoreUploader = new PicGoCoreUploader(this.settings);
                        if (this.settings.uploader === "PicGo") {
                            this.uploader = this.picGoUploader;
                        }
                        else if (this.settings.uploader === "PicGo-Core") {
                            this.uploader = this.picGoCoreUploader;
                        }
                        else {
                            new obsidian.Notice("unknown uploader");
                        }
                        obsidian.addIcon("upload", "<svg t=\"1636630783429\" class=\"icon\" viewBox=\"0 0 100 100\" version=\"1.1\" p-id=\"4649\" xmlns=\"http://www.w3.org/2000/svg\">\n      <path d=\"M 71.638 35.336 L 79.408 35.336 C 83.7 35.336 87.178 38.662 87.178 42.765 L 87.178 84.864 C 87.178 88.969 83.7 92.295 79.408 92.295 L 17.249 92.295 C 12.957 92.295 9.479 88.969 9.479 84.864 L 9.479 42.765 C 9.479 38.662 12.957 35.336 17.249 35.336 L 25.019 35.336 L 25.019 42.765 L 17.249 42.765 L 17.249 84.864 L 79.408 84.864 L 79.408 42.765 L 71.638 42.765 L 71.638 35.336 Z M 49.014 10.179 L 67.326 27.688 L 61.835 32.942 L 52.849 24.352 L 52.849 59.731 L 45.078 59.731 L 45.078 24.455 L 36.194 32.947 L 30.702 27.692 L 49.012 10.181 Z\" p-id=\"4650\" fill=\"#8a8a8a\"></path>\n    </svg>");
                        this.addSettingTab(new SettingTab(this.app, this));
                        this.addCommand({
                            id: "upload all images",
                            name: "upload all images",
                            checkCallback: function (checking) {
                                var leaf = _this.app.workspace.activeLeaf;
                                if (leaf) {
                                    if (!checking) {
                                        _this.uploadAllFile();
                                    }
                                    return true;
                                }
                                return false;
                            },
                        });
                        this.addCommand({
                            id: "doanload all images",
                            name: "download all images",
                            checkCallback: function (checking) {
                                var leaf = _this.app.workspace.activeLeaf;
                                if (leaf) {
                                    if (!checking) {
                                        _this.downloadAllImageFiles();
                                    }
                                    return true;
                                }
                                return false;
                            },
                        });
                        this.setupPasteHandler();
                        this.registerFileMenu();
                        return [2 /*return*/];
                }
            });
        });
    };
    imageAutoUploadPlugin.prototype.downloadAllImageFiles = function () {
        return __awaiter(this, void 0, void 0, function () {
            var folderPath, fileArray, imageArray, fileArray_1, fileArray_1_1, file, url, asset, _a, name_1, ext, response, activeFolder, basePath, abstractActiveFolder, e_1_1, value;
            var e_1, _b;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        folderPath = this.getFileAssetPath();
                        fileArray = this.helper.getAllFiles();
                        if (!fs.existsSync(folderPath)) {
                            fs.mkdirSync(folderPath);
                        }
                        imageArray = [];
                        _c.label = 1;
                    case 1:
                        _c.trys.push([1, 6, 7, 8]);
                        fileArray_1 = __values(fileArray), fileArray_1_1 = fileArray_1.next();
                        _c.label = 2;
                    case 2:
                        if (!!fileArray_1_1.done) return [3 /*break*/, 5];
                        file = fileArray_1_1.value;
                        if (!file.path.startsWith("http")) {
                            return [3 /*break*/, 4];
                        }
                        url = file.path;
                        asset = getUrlAsset(url);
                        if (!isAnImage(asset.substr(asset.lastIndexOf(".")))) {
                            return [3 /*break*/, 4];
                        }
                        _a = __read([
                            decodeURI(path.parse(asset).name).replaceAll(/[\\\\/:*?\"<>|]/g, "-"),
                            path.parse(asset).ext,
                        ], 2), name_1 = _a[0], ext = _a[1];
                        // 如果文件名已存在，则用随机值替换
                        if (fs.existsSync(path.join(folderPath, encodeURI(asset)))) {
                            name_1 = (Math.random() + 1).toString(36).substr(2, 5);
                        }
                        name_1 = "image-" + name_1;
                        return [4 /*yield*/, this.download(url, path.join(folderPath, "" + name_1 + ext))];
                    case 3:
                        response = _c.sent();
                        if (response.ok) {
                            activeFolder = this.app.vault.getAbstractFileByPath(this.app.workspace.getActiveFile().path).parent.path;
                            basePath = this.app.vault.adapter.getBasePath();
                            abstractActiveFolder = path.resolve(basePath, activeFolder);
                            imageArray.push({
                                source: file.source,
                                name: name_1,
                                path: obsidian.normalizePath(path.relative(abstractActiveFolder, response.path)),
                            });
                        }
                        _c.label = 4;
                    case 4:
                        fileArray_1_1 = fileArray_1.next();
                        return [3 /*break*/, 2];
                    case 5: return [3 /*break*/, 8];
                    case 6:
                        e_1_1 = _c.sent();
                        e_1 = { error: e_1_1 };
                        return [3 /*break*/, 8];
                    case 7:
                        try {
                            if (fileArray_1_1 && !fileArray_1_1.done && (_b = fileArray_1.return)) _b.call(fileArray_1);
                        }
                        finally { if (e_1) throw e_1.error; }
                        return [7 /*endfinally*/];
                    case 8:
                        value = this.helper.getValue();
                        imageArray.map(function (image) {
                            value = value.replace(image.source, "![" + image.name + "](" + encodeURI(image.path) + ")");
                        });
                        this.helper.setValue(value);
                        new obsidian.Notice("all: " + fileArray.length + "\nsuccess: " + imageArray.length + "\nfailed: " + (fileArray.length - imageArray.length));
                        return [2 /*return*/];
                }
            });
        });
    };
    // 获取当前文件所属的附件文件夹
    imageAutoUploadPlugin.prototype.getFileAssetPath = function () {
        var basePath = this.app.vault.adapter.getBasePath();
        // @ts-ignore
        var assetFolder = this.app.vault.config.attachmentFolderPath;
        var activeFile = this.app.vault.getAbstractFileByPath(this.app.workspace.getActiveFile().path);
        // 当前文件夹下的子文件夹
        if (assetFolder.startsWith("./")) {
            var activeFolder = decodeURI(path.resolve(basePath, activeFile.parent.path));
            return path.join(activeFolder, assetFolder);
        }
        else {
            // 根文件夹
            return path.join(basePath, assetFolder);
        }
    };
    imageAutoUploadPlugin.prototype.download = function (url, path) {
        return __awaiter(this, void 0, void 0, function () {
            var response, buffer;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, obsidian.requestUrl({ url: url })];
                    case 1:
                        response = _a.sent();
                        if (response.status !== 200) {
                            return [2 /*return*/, {
                                    ok: false,
                                    msg: "error",
                                }];
                        }
                        buffer = Buffer.from(response.arrayBuffer);
                        try {
                            fs.writeFileSync(path, buffer);
                            return [2 /*return*/, {
                                    ok: true,
                                    msg: "ok",
                                    path: path,
                                }];
                        }
                        catch (err) {
                            console.error(err);
                            return [2 /*return*/, {
                                    ok: false,
                                    msg: err,
                                }];
                        }
                        return [2 /*return*/];
                }
            });
        });
    };
    imageAutoUploadPlugin.prototype.registerFileMenu = function () {
        var _this = this;
        this.registerEvent(this.app.workspace.on("file-menu", function (menu, file, source) {
            if (!isAssetTypeAnImage(file.path)) {
                return false;
            }
            menu.addItem(function (item) {
                item
                    .setTitle("Upload")
                    .setIcon("upload")
                    .onClick(function () {
                    if (!(file instanceof obsidian.TFile)) {
                        return false;
                    }
                    var basePath = _this.app.vault.adapter.getBasePath();
                    var uri = decodeURI(path.resolve(basePath, file.path));
                    _this.uploader.uploadFiles([uri]).then(function (res) {
                        if (res.success) {
                            // @ts-ignore
                            var uploadUrl = res.result[0];
                            var sourceUri = encodeURI(path.relative(_this.app.workspace.getActiveFile().parent.path, file.path).replaceAll("\\", "/"));
                            var value = _this.helper.getValue();
                            var menuMode = _this.settings.menuMode;
                            if (menuMode === "auto") {
                                // @ts-ignore
                                menuMode = _this.app.vault.config.newLinkFormat;
                            }
                            if (menuMode === "relative") {
                                // 替换相对路径的 ![]()格式
                                value = value.replaceAll(sourceUri, uploadUrl);
                                // 替换相对路径的 ![[]]格式
                                value = value.replaceAll("![[" + decodeURI(sourceUri) + "]]", "![](" + uploadUrl + ")");
                            }
                            else if (menuMode === "absolute") {
                                // 替换绝对路径的 ![[]]格式
                                value = value.replaceAll("![[" + file.path + "]]", "![](" + uploadUrl + ")");
                                // 替换绝对路径的 ![]()格式
                                value = value.replaceAll(file.path.replaceAll(" ", "%20"), uploadUrl);
                            }
                            else {
                                new obsidian.Notice("Not support " + menuMode + " mode");
                            }
                            _this.helper.setValue(value);
                        }
                        else {
                            new obsidian.Notice(res.msg || "Upload error");
                        }
                    });
                });
            });
        }));
    };
    // uploda all file
    imageAutoUploadPlugin.prototype.uploadAllFile = function () {
        var e_2, _a;
        var _this = this;
        var key = this.helper.getValue();
        var thisPath = this.app.vault.getAbstractFileByPath(this.app.workspace.getActiveFile().path);
        var basePath = this.app.vault.adapter.getBasePath();
        var imageList = [];
        var fileArray = this.helper.getAllFiles();
        try {
            for (var fileArray_2 = __values(fileArray), fileArray_2_1 = fileArray_2.next(); !fileArray_2_1.done; fileArray_2_1 = fileArray_2.next()) {
                var match = fileArray_2_1.value;
                var imageName = match.name;
                var encodedUri = match.path;
                if (!encodedUri.startsWith("http")) {
                    var abstractImageFile = void 0;
                    // 当路径以“.”开头时，识别为相对路径，不然就认为时绝对路径
                    if (encodedUri.startsWith(".")) {
                        abstractImageFile = decodeURI(path.join(basePath, path.posix.resolve(path.posix.join("/", thisPath.parent.path), encodedUri)));
                    }
                    else {
                        abstractImageFile = decodeURI(path.join(basePath, encodedUri));
                        // 当解析为绝对路径却找不到文件，尝试解析为相对路径
                        if (!fs.existsSync(abstractImageFile)) {
                            abstractImageFile = decodeURI(path.join(basePath, path.posix.resolve(path.posix.join("/", thisPath.parent.path), encodedUri)));
                        }
                    }
                    if (fs.existsSync(abstractImageFile) &&
                        isAssetTypeAnImage(abstractImageFile)) {
                        imageList.push({
                            path: abstractImageFile,
                            name: imageName,
                            source: match.source,
                        });
                    }
                }
            }
        }
        catch (e_2_1) { e_2 = { error: e_2_1 }; }
        finally {
            try {
                if (fileArray_2_1 && !fileArray_2_1.done && (_a = fileArray_2.return)) _a.call(fileArray_2);
            }
            finally { if (e_2) throw e_2.error; }
        }
        if (imageList.length === 0) {
            new obsidian.Notice("没有解析到图像文件");
            return;
        }
        else {
            new obsidian.Notice("\u5171\u627E\u5230" + imageList.length + "\u4E2A\u56FE\u50CF\u6587\u4EF6\uFF0C\u5F00\u59CB\u4E0A\u4F20");
        }
        console.log(imageList);
        this.uploader.uploadFiles(imageList.map(function (item) { return item.path; })).then(function (res) {
            if (res.success) {
                var uploadUrlList_1 = res.result;
                imageList.map(function (item) {
                    // gitea不能上传超过1M的数据，上传多张照片，错误的话会返回什么？还有待验证
                    var uploadImage = uploadUrlList_1.shift();
                    key = key.replaceAll(item.source, "![" + item.name + "](" + uploadImage + ")");
                });
                _this.helper.setValue(key);
            }
            else {
                new obsidian.Notice("Upload error");
            }
        });
    };
    imageAutoUploadPlugin.prototype.setupPasteHandler = function () {
        var _this = this;
        this.registerEvent(this.app.workspace.on("editor-paste", function (evt, editor, markdownView) {
            var _a;
            var allowUpload = _this.helper.getFrontmatterValue("image-auto-upload", _this.settings.uploadByClipSwitch);
            var files = evt.clipboardData.files;
            if (!allowUpload) {
                return;
            }
            // 剪贴板内容有md格式的图片时
            if (_this.settings.workOnNetWork) {
                var clipboardValue = evt.clipboardData.getData("text/plain");
                var imageList_1 = _this.helper
                    .getImageLink(clipboardValue)
                    .filter(function (image) { return image.path.startsWith("http"); });
                if (imageList_1.length !== 0) {
                    _this.uploader
                        .uploadFiles(imageList_1.map(function (item) { return item.path; }))
                        .then(function (res) {
                        var value = _this.helper.getValue();
                        if (res.success) {
                            var uploadUrlList_2 = res.result;
                            imageList_1.map(function (item) {
                                var uploadImage = uploadUrlList_2.shift();
                                value = value.replaceAll(item.source, "![" + item.name + "](" + uploadImage + ")");
                            });
                            _this.helper.setValue(value);
                        }
                        else {
                            new obsidian.Notice("Upload error");
                        }
                    });
                }
            }
            // 剪贴板中是图片时进行上传
            if (isCopyImageFile() ||
                files.length !== 0 ||
                ((_a = files[0]) === null || _a === void 0 ? void 0 : _a.type.startsWith("image"))) {
                _this.uploadFileAndEmbedImgurImage(editor, function (editor, pasteId) { return __awaiter(_this, void 0, void 0, function () {
                    var res, url;
                    return __generator(this, function (_a) {
                        switch (_a.label) {
                            case 0: return [4 /*yield*/, this.uploader.uploadFileByClipboard()];
                            case 1:
                                res = _a.sent();
                                if (res.code !== 0) {
                                    this.handleFailedUpload(editor, pasteId, res.msg);
                                    return [2 /*return*/];
                                }
                                url = res.data;
                                return [2 /*return*/, url];
                        }
                    });
                }); }).catch(console.error);
                evt.preventDefault();
            }
        }));
        this.registerEvent(this.app.workspace.on("editor-drop", function (evt, editor, markdownView) { return __awaiter(_this, void 0, void 0, function () {
            var allowUpload, files, sendFiles_1, files_1, data;
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        allowUpload = this.helper.getFrontmatterValue("image-auto-upload", this.settings.uploadByClipSwitch);
                        files = evt.dataTransfer.files;
                        if (!allowUpload) {
                            return [2 /*return*/];
                        }
                        if (!(files.length !== 0 && files[0].type.startsWith("image"))) return [3 /*break*/, 2];
                        sendFiles_1 = [];
                        files_1 = evt.dataTransfer.files;
                        Array.from(files_1).forEach(function (item, index) {
                            sendFiles_1.push(item.path);
                        });
                        evt.preventDefault();
                        return [4 /*yield*/, this.uploader.uploadFiles(sendFiles_1)];
                    case 1:
                        data = _a.sent();
                        if (data.success) {
                            data.result.map(function (value) {
                                var pasteId = (Math.random() + 1).toString(36).substr(2, 5);
                                _this.insertTemporaryText(editor, pasteId);
                                _this.embedMarkDownImage(editor, pasteId, value);
                            });
                        }
                        else {
                            new obsidian.Notice("Upload error");
                        }
                        _a.label = 2;
                    case 2: return [2 /*return*/];
                }
            });
        }); }));
    };
    imageAutoUploadPlugin.prototype.uploadFileAndEmbedImgurImage = function (editor, callback) {
        return __awaiter(this, void 0, void 0, function () {
            var pasteId, url, e_3;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        pasteId = (Math.random() + 1).toString(36).substr(2, 5);
                        this.insertTemporaryText(editor, pasteId);
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 3, , 4]);
                        return [4 /*yield*/, callback(editor, pasteId)];
                    case 2:
                        url = _a.sent();
                        this.embedMarkDownImage(editor, pasteId, url);
                        return [3 /*break*/, 4];
                    case 3:
                        e_3 = _a.sent();
                        this.handleFailedUpload(editor, pasteId, e_3);
                        return [3 /*break*/, 4];
                    case 4: return [2 /*return*/];
                }
            });
        });
    };
    imageAutoUploadPlugin.prototype.insertTemporaryText = function (editor, pasteId) {
        var progressText = imageAutoUploadPlugin.progressTextFor(pasteId);
        editor.replaceSelection(progressText + "\n");
    };
    imageAutoUploadPlugin.progressTextFor = function (id) {
        return "![Uploading file..." + id + "]()";
    };
    imageAutoUploadPlugin.prototype.embedMarkDownImage = function (editor, pasteId, imageUrl) {
        var progressText = imageAutoUploadPlugin.progressTextFor(pasteId);
        var markDownImage = "![](" + imageUrl + ")";
        imageAutoUploadPlugin.replaceFirstOccurrence(editor, progressText, markDownImage);
    };
    imageAutoUploadPlugin.prototype.handleFailedUpload = function (editor, pasteId, reason) {
        console.error("Failed request: ", reason);
        var progressText = imageAutoUploadPlugin.progressTextFor(pasteId);
        imageAutoUploadPlugin.replaceFirstOccurrence(editor, progressText, "⚠️upload failed, check dev console");
    };
    imageAutoUploadPlugin.replaceFirstOccurrence = function (editor, target, replacement) {
        var lines = editor.getValue().split("\n");
        for (var i = 0; i < lines.length; i++) {
            var ch = lines[i].indexOf(target);
            if (ch != -1) {
                var from = { line: i, ch: ch };
                var to = { line: i, ch: ch + target.length };
                editor.replaceRange(replacement, from, to);
                break;
            }
        }
    };
    return imageAutoUploadPlugin;
}(obsidian.Plugin));

module.exports = imageAutoUploadPlugin;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWFpbi5qcyIsInNvdXJjZXMiOlsibm9kZV9tb2R1bGVzL3RzbGliL3RzbGliLmVzNi5qcyIsInNyYy91dGlscy50cyIsInNyYy91cGxvYWRlci50cyIsInNyYy9oZWxwZXIudHMiLCJzcmMvc2V0dGluZy50cyIsInNyYy9tYWluLnRzIl0sInNvdXJjZXNDb250ZW50IjpbIi8qISAqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxyXG5Db3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi5cclxuXHJcblBlcm1pc3Npb24gdG8gdXNlLCBjb3B5LCBtb2RpZnksIGFuZC9vciBkaXN0cmlidXRlIHRoaXMgc29mdHdhcmUgZm9yIGFueVxyXG5wdXJwb3NlIHdpdGggb3Igd2l0aG91dCBmZWUgaXMgaGVyZWJ5IGdyYW50ZWQuXHJcblxyXG5USEUgU09GVFdBUkUgSVMgUFJPVklERUQgXCJBUyBJU1wiIEFORCBUSEUgQVVUSE9SIERJU0NMQUlNUyBBTEwgV0FSUkFOVElFUyBXSVRIXHJcblJFR0FSRCBUTyBUSElTIFNPRlRXQVJFIElOQ0xVRElORyBBTEwgSU1QTElFRCBXQVJSQU5USUVTIE9GIE1FUkNIQU5UQUJJTElUWVxyXG5BTkQgRklUTkVTUy4gSU4gTk8gRVZFTlQgU0hBTEwgVEhFIEFVVEhPUiBCRSBMSUFCTEUgRk9SIEFOWSBTUEVDSUFMLCBESVJFQ1QsXHJcbklORElSRUNULCBPUiBDT05TRVFVRU5USUFMIERBTUFHRVMgT1IgQU5ZIERBTUFHRVMgV0hBVFNPRVZFUiBSRVNVTFRJTkcgRlJPTVxyXG5MT1NTIE9GIFVTRSwgREFUQSBPUiBQUk9GSVRTLCBXSEVUSEVSIElOIEFOIEFDVElPTiBPRiBDT05UUkFDVCwgTkVHTElHRU5DRSBPUlxyXG5PVEhFUiBUT1JUSU9VUyBBQ1RJT04sIEFSSVNJTkcgT1VUIE9GIE9SIElOIENPTk5FQ1RJT04gV0lUSCBUSEUgVVNFIE9SXHJcblBFUkZPUk1BTkNFIE9GIFRISVMgU09GVFdBUkUuXHJcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqICovXHJcbi8qIGdsb2JhbCBSZWZsZWN0LCBQcm9taXNlICovXHJcblxyXG52YXIgZXh0ZW5kU3RhdGljcyA9IGZ1bmN0aW9uKGQsIGIpIHtcclxuICAgIGV4dGVuZFN0YXRpY3MgPSBPYmplY3Quc2V0UHJvdG90eXBlT2YgfHxcclxuICAgICAgICAoeyBfX3Byb3RvX186IFtdIH0gaW5zdGFuY2VvZiBBcnJheSAmJiBmdW5jdGlvbiAoZCwgYikgeyBkLl9fcHJvdG9fXyA9IGI7IH0pIHx8XHJcbiAgICAgICAgZnVuY3Rpb24gKGQsIGIpIHsgZm9yICh2YXIgcCBpbiBiKSBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKGIsIHApKSBkW3BdID0gYltwXTsgfTtcclxuICAgIHJldHVybiBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG59O1xyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZXh0ZW5kcyhkLCBiKSB7XHJcbiAgICBpZiAodHlwZW9mIGIgIT09IFwiZnVuY3Rpb25cIiAmJiBiICE9PSBudWxsKVxyXG4gICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoXCJDbGFzcyBleHRlbmRzIHZhbHVlIFwiICsgU3RyaW5nKGIpICsgXCIgaXMgbm90IGEgY29uc3RydWN0b3Igb3IgbnVsbFwiKTtcclxuICAgIGV4dGVuZFN0YXRpY3MoZCwgYik7XHJcbiAgICBmdW5jdGlvbiBfXygpIHsgdGhpcy5jb25zdHJ1Y3RvciA9IGQ7IH1cclxuICAgIGQucHJvdG90eXBlID0gYiA9PT0gbnVsbCA/IE9iamVjdC5jcmVhdGUoYikgOiAoX18ucHJvdG90eXBlID0gYi5wcm90b3R5cGUsIG5ldyBfXygpKTtcclxufVxyXG5cclxuZXhwb3J0IHZhciBfX2Fzc2lnbiA9IGZ1bmN0aW9uKCkge1xyXG4gICAgX19hc3NpZ24gPSBPYmplY3QuYXNzaWduIHx8IGZ1bmN0aW9uIF9fYXNzaWduKHQpIHtcclxuICAgICAgICBmb3IgKHZhciBzLCBpID0gMSwgbiA9IGFyZ3VtZW50cy5sZW5ndGg7IGkgPCBuOyBpKyspIHtcclxuICAgICAgICAgICAgcyA9IGFyZ3VtZW50c1tpXTtcclxuICAgICAgICAgICAgZm9yICh2YXIgcCBpbiBzKSBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHMsIHApKSB0W3BdID0gc1twXTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHQ7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gX19hc3NpZ24uYXBwbHkodGhpcywgYXJndW1lbnRzKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fcmVzdChzLCBlKSB7XHJcbiAgICB2YXIgdCA9IHt9O1xyXG4gICAgZm9yICh2YXIgcCBpbiBzKSBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHMsIHApICYmIGUuaW5kZXhPZihwKSA8IDApXHJcbiAgICAgICAgdFtwXSA9IHNbcF07XHJcbiAgICBpZiAocyAhPSBudWxsICYmIHR5cGVvZiBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzID09PSBcImZ1bmN0aW9uXCIpXHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDAsIHAgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzKHMpOyBpIDwgcC5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICBpZiAoZS5pbmRleE9mKHBbaV0pIDwgMCAmJiBPYmplY3QucHJvdG90eXBlLnByb3BlcnR5SXNFbnVtZXJhYmxlLmNhbGwocywgcFtpXSkpXHJcbiAgICAgICAgICAgICAgICB0W3BbaV1dID0gc1twW2ldXTtcclxuICAgICAgICB9XHJcbiAgICByZXR1cm4gdDtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZGVjb3JhdGUoZGVjb3JhdG9ycywgdGFyZ2V0LCBrZXksIGRlc2MpIHtcclxuICAgIHZhciBjID0gYXJndW1lbnRzLmxlbmd0aCwgciA9IGMgPCAzID8gdGFyZ2V0IDogZGVzYyA9PT0gbnVsbCA/IGRlc2MgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKHRhcmdldCwga2V5KSA6IGRlc2MsIGQ7XHJcbiAgICBpZiAodHlwZW9mIFJlZmxlY3QgPT09IFwib2JqZWN0XCIgJiYgdHlwZW9mIFJlZmxlY3QuZGVjb3JhdGUgPT09IFwiZnVuY3Rpb25cIikgciA9IFJlZmxlY3QuZGVjb3JhdGUoZGVjb3JhdG9ycywgdGFyZ2V0LCBrZXksIGRlc2MpO1xyXG4gICAgZWxzZSBmb3IgKHZhciBpID0gZGVjb3JhdG9ycy5sZW5ndGggLSAxOyBpID49IDA7IGktLSkgaWYgKGQgPSBkZWNvcmF0b3JzW2ldKSByID0gKGMgPCAzID8gZChyKSA6IGMgPiAzID8gZCh0YXJnZXQsIGtleSwgcikgOiBkKHRhcmdldCwga2V5KSkgfHwgcjtcclxuICAgIHJldHVybiBjID4gMyAmJiByICYmIE9iamVjdC5kZWZpbmVQcm9wZXJ0eSh0YXJnZXQsIGtleSwgciksIHI7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3BhcmFtKHBhcmFtSW5kZXgsIGRlY29yYXRvcikge1xyXG4gICAgcmV0dXJuIGZ1bmN0aW9uICh0YXJnZXQsIGtleSkgeyBkZWNvcmF0b3IodGFyZ2V0LCBrZXksIHBhcmFtSW5kZXgpOyB9XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX21ldGFkYXRhKG1ldGFkYXRhS2V5LCBtZXRhZGF0YVZhbHVlKSB7XHJcbiAgICBpZiAodHlwZW9mIFJlZmxlY3QgPT09IFwib2JqZWN0XCIgJiYgdHlwZW9mIFJlZmxlY3QubWV0YWRhdGEgPT09IFwiZnVuY3Rpb25cIikgcmV0dXJuIFJlZmxlY3QubWV0YWRhdGEobWV0YWRhdGFLZXksIG1ldGFkYXRhVmFsdWUpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hd2FpdGVyKHRoaXNBcmcsIF9hcmd1bWVudHMsIFAsIGdlbmVyYXRvcikge1xyXG4gICAgZnVuY3Rpb24gYWRvcHQodmFsdWUpIHsgcmV0dXJuIHZhbHVlIGluc3RhbmNlb2YgUCA/IHZhbHVlIDogbmV3IFAoZnVuY3Rpb24gKHJlc29sdmUpIHsgcmVzb2x2ZSh2YWx1ZSk7IH0pOyB9XHJcbiAgICByZXR1cm4gbmV3IChQIHx8IChQID0gUHJvbWlzZSkpKGZ1bmN0aW9uIChyZXNvbHZlLCByZWplY3QpIHtcclxuICAgICAgICBmdW5jdGlvbiBmdWxmaWxsZWQodmFsdWUpIHsgdHJ5IHsgc3RlcChnZW5lcmF0b3IubmV4dCh2YWx1ZSkpOyB9IGNhdGNoIChlKSB7IHJlamVjdChlKTsgfSB9XHJcbiAgICAgICAgZnVuY3Rpb24gcmVqZWN0ZWQodmFsdWUpIHsgdHJ5IHsgc3RlcChnZW5lcmF0b3JbXCJ0aHJvd1wiXSh2YWx1ZSkpOyB9IGNhdGNoIChlKSB7IHJlamVjdChlKTsgfSB9XHJcbiAgICAgICAgZnVuY3Rpb24gc3RlcChyZXN1bHQpIHsgcmVzdWx0LmRvbmUgPyByZXNvbHZlKHJlc3VsdC52YWx1ZSkgOiBhZG9wdChyZXN1bHQudmFsdWUpLnRoZW4oZnVsZmlsbGVkLCByZWplY3RlZCk7IH1cclxuICAgICAgICBzdGVwKChnZW5lcmF0b3IgPSBnZW5lcmF0b3IuYXBwbHkodGhpc0FyZywgX2FyZ3VtZW50cyB8fCBbXSkpLm5leHQoKSk7XHJcbiAgICB9KTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZ2VuZXJhdG9yKHRoaXNBcmcsIGJvZHkpIHtcclxuICAgIHZhciBfID0geyBsYWJlbDogMCwgc2VudDogZnVuY3Rpb24oKSB7IGlmICh0WzBdICYgMSkgdGhyb3cgdFsxXTsgcmV0dXJuIHRbMV07IH0sIHRyeXM6IFtdLCBvcHM6IFtdIH0sIGYsIHksIHQsIGc7XHJcbiAgICByZXR1cm4gZyA9IHsgbmV4dDogdmVyYigwKSwgXCJ0aHJvd1wiOiB2ZXJiKDEpLCBcInJldHVyblwiOiB2ZXJiKDIpIH0sIHR5cGVvZiBTeW1ib2wgPT09IFwiZnVuY3Rpb25cIiAmJiAoZ1tTeW1ib2wuaXRlcmF0b3JdID0gZnVuY3Rpb24oKSB7IHJldHVybiB0aGlzOyB9KSwgZztcclxuICAgIGZ1bmN0aW9uIHZlcmIobikgeyByZXR1cm4gZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHN0ZXAoW24sIHZdKTsgfTsgfVxyXG4gICAgZnVuY3Rpb24gc3RlcChvcCkge1xyXG4gICAgICAgIGlmIChmKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiR2VuZXJhdG9yIGlzIGFscmVhZHkgZXhlY3V0aW5nLlwiKTtcclxuICAgICAgICB3aGlsZSAoXykgdHJ5IHtcclxuICAgICAgICAgICAgaWYgKGYgPSAxLCB5ICYmICh0ID0gb3BbMF0gJiAyID8geVtcInJldHVyblwiXSA6IG9wWzBdID8geVtcInRocm93XCJdIHx8ICgodCA9IHlbXCJyZXR1cm5cIl0pICYmIHQuY2FsbCh5KSwgMCkgOiB5Lm5leHQpICYmICEodCA9IHQuY2FsbCh5LCBvcFsxXSkpLmRvbmUpIHJldHVybiB0O1xyXG4gICAgICAgICAgICBpZiAoeSA9IDAsIHQpIG9wID0gW29wWzBdICYgMiwgdC52YWx1ZV07XHJcbiAgICAgICAgICAgIHN3aXRjaCAob3BbMF0pIHtcclxuICAgICAgICAgICAgICAgIGNhc2UgMDogY2FzZSAxOiB0ID0gb3A7IGJyZWFrO1xyXG4gICAgICAgICAgICAgICAgY2FzZSA0OiBfLmxhYmVsKys7IHJldHVybiB7IHZhbHVlOiBvcFsxXSwgZG9uZTogZmFsc2UgfTtcclxuICAgICAgICAgICAgICAgIGNhc2UgNTogXy5sYWJlbCsrOyB5ID0gb3BbMV07IG9wID0gWzBdOyBjb250aW51ZTtcclxuICAgICAgICAgICAgICAgIGNhc2UgNzogb3AgPSBfLm9wcy5wb3AoKTsgXy50cnlzLnBvcCgpOyBjb250aW51ZTtcclxuICAgICAgICAgICAgICAgIGRlZmF1bHQ6XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKCEodCA9IF8udHJ5cywgdCA9IHQubGVuZ3RoID4gMCAmJiB0W3QubGVuZ3RoIC0gMV0pICYmIChvcFswXSA9PT0gNiB8fCBvcFswXSA9PT0gMikpIHsgXyA9IDA7IGNvbnRpbnVlOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9wWzBdID09PSAzICYmICghdCB8fCAob3BbMV0gPiB0WzBdICYmIG9wWzFdIDwgdFszXSkpKSB7IF8ubGFiZWwgPSBvcFsxXTsgYnJlYWs7IH1cclxuICAgICAgICAgICAgICAgICAgICBpZiAob3BbMF0gPT09IDYgJiYgXy5sYWJlbCA8IHRbMV0pIHsgXy5sYWJlbCA9IHRbMV07IHQgPSBvcDsgYnJlYWs7IH1cclxuICAgICAgICAgICAgICAgICAgICBpZiAodCAmJiBfLmxhYmVsIDwgdFsyXSkgeyBfLmxhYmVsID0gdFsyXTsgXy5vcHMucHVzaChvcCk7IGJyZWFrOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHRbMl0pIF8ub3BzLnBvcCgpO1xyXG4gICAgICAgICAgICAgICAgICAgIF8udHJ5cy5wb3AoKTsgY29udGludWU7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgb3AgPSBib2R5LmNhbGwodGhpc0FyZywgXyk7XHJcbiAgICAgICAgfSBjYXRjaCAoZSkgeyBvcCA9IFs2LCBlXTsgeSA9IDA7IH0gZmluYWxseSB7IGYgPSB0ID0gMDsgfVxyXG4gICAgICAgIGlmIChvcFswXSAmIDUpIHRocm93IG9wWzFdOyByZXR1cm4geyB2YWx1ZTogb3BbMF0gPyBvcFsxXSA6IHZvaWQgMCwgZG9uZTogdHJ1ZSB9O1xyXG4gICAgfVxyXG59XHJcblxyXG5leHBvcnQgdmFyIF9fY3JlYXRlQmluZGluZyA9IE9iamVjdC5jcmVhdGUgPyAoZnVuY3Rpb24obywgbSwgaywgazIpIHtcclxuICAgIGlmIChrMiA9PT0gdW5kZWZpbmVkKSBrMiA9IGs7XHJcbiAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkobywgazIsIHsgZW51bWVyYWJsZTogdHJ1ZSwgZ2V0OiBmdW5jdGlvbigpIHsgcmV0dXJuIG1ba107IH0gfSk7XHJcbn0pIDogKGZ1bmN0aW9uKG8sIG0sIGssIGsyKSB7XHJcbiAgICBpZiAoazIgPT09IHVuZGVmaW5lZCkgazIgPSBrO1xyXG4gICAgb1trMl0gPSBtW2tdO1xyXG59KTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2V4cG9ydFN0YXIobSwgbykge1xyXG4gICAgZm9yICh2YXIgcCBpbiBtKSBpZiAocCAhPT0gXCJkZWZhdWx0XCIgJiYgIU9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvLCBwKSkgX19jcmVhdGVCaW5kaW5nKG8sIG0sIHApO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX192YWx1ZXMobykge1xyXG4gICAgdmFyIHMgPSB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgU3ltYm9sLml0ZXJhdG9yLCBtID0gcyAmJiBvW3NdLCBpID0gMDtcclxuICAgIGlmIChtKSByZXR1cm4gbS5jYWxsKG8pO1xyXG4gICAgaWYgKG8gJiYgdHlwZW9mIG8ubGVuZ3RoID09PSBcIm51bWJlclwiKSByZXR1cm4ge1xyXG4gICAgICAgIG5leHQ6IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgaWYgKG8gJiYgaSA+PSBvLmxlbmd0aCkgbyA9IHZvaWQgMDtcclxuICAgICAgICAgICAgcmV0dXJuIHsgdmFsdWU6IG8gJiYgb1tpKytdLCBkb25lOiAhbyB9O1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB0aHJvdyBuZXcgVHlwZUVycm9yKHMgPyBcIk9iamVjdCBpcyBub3QgaXRlcmFibGUuXCIgOiBcIlN5bWJvbC5pdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3JlYWQobywgbikge1xyXG4gICAgdmFyIG0gPSB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgb1tTeW1ib2wuaXRlcmF0b3JdO1xyXG4gICAgaWYgKCFtKSByZXR1cm4gbztcclxuICAgIHZhciBpID0gbS5jYWxsKG8pLCByLCBhciA9IFtdLCBlO1xyXG4gICAgdHJ5IHtcclxuICAgICAgICB3aGlsZSAoKG4gPT09IHZvaWQgMCB8fCBuLS0gPiAwKSAmJiAhKHIgPSBpLm5leHQoKSkuZG9uZSkgYXIucHVzaChyLnZhbHVlKTtcclxuICAgIH1cclxuICAgIGNhdGNoIChlcnJvcikgeyBlID0geyBlcnJvcjogZXJyb3IgfTsgfVxyXG4gICAgZmluYWxseSB7XHJcbiAgICAgICAgdHJ5IHtcclxuICAgICAgICAgICAgaWYgKHIgJiYgIXIuZG9uZSAmJiAobSA9IGlbXCJyZXR1cm5cIl0pKSBtLmNhbGwoaSk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGZpbmFsbHkgeyBpZiAoZSkgdGhyb3cgZS5lcnJvcjsgfVxyXG4gICAgfVxyXG4gICAgcmV0dXJuIGFyO1xyXG59XHJcblxyXG4vKiogQGRlcHJlY2F0ZWQgKi9cclxuZXhwb3J0IGZ1bmN0aW9uIF9fc3ByZWFkKCkge1xyXG4gICAgZm9yICh2YXIgYXIgPSBbXSwgaSA9IDA7IGkgPCBhcmd1bWVudHMubGVuZ3RoOyBpKyspXHJcbiAgICAgICAgYXIgPSBhci5jb25jYXQoX19yZWFkKGFyZ3VtZW50c1tpXSkpO1xyXG4gICAgcmV0dXJuIGFyO1xyXG59XHJcblxyXG4vKiogQGRlcHJlY2F0ZWQgKi9cclxuZXhwb3J0IGZ1bmN0aW9uIF9fc3ByZWFkQXJyYXlzKCkge1xyXG4gICAgZm9yICh2YXIgcyA9IDAsIGkgPSAwLCBpbCA9IGFyZ3VtZW50cy5sZW5ndGg7IGkgPCBpbDsgaSsrKSBzICs9IGFyZ3VtZW50c1tpXS5sZW5ndGg7XHJcbiAgICBmb3IgKHZhciByID0gQXJyYXkocyksIGsgPSAwLCBpID0gMDsgaSA8IGlsOyBpKyspXHJcbiAgICAgICAgZm9yICh2YXIgYSA9IGFyZ3VtZW50c1tpXSwgaiA9IDAsIGpsID0gYS5sZW5ndGg7IGogPCBqbDsgaisrLCBrKyspXHJcbiAgICAgICAgICAgIHJba10gPSBhW2pdO1xyXG4gICAgcmV0dXJuIHI7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3NwcmVhZEFycmF5KHRvLCBmcm9tKSB7XHJcbiAgICBmb3IgKHZhciBpID0gMCwgaWwgPSBmcm9tLmxlbmd0aCwgaiA9IHRvLmxlbmd0aDsgaSA8IGlsOyBpKyssIGorKylcclxuICAgICAgICB0b1tqXSA9IGZyb21baV07XHJcbiAgICByZXR1cm4gdG87XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2F3YWl0KHYpIHtcclxuICAgIHJldHVybiB0aGlzIGluc3RhbmNlb2YgX19hd2FpdCA/ICh0aGlzLnYgPSB2LCB0aGlzKSA6IG5ldyBfX2F3YWl0KHYpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hc3luY0dlbmVyYXRvcih0aGlzQXJnLCBfYXJndW1lbnRzLCBnZW5lcmF0b3IpIHtcclxuICAgIGlmICghU3ltYm9sLmFzeW5jSXRlcmF0b3IpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJTeW1ib2wuYXN5bmNJdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbiAgICB2YXIgZyA9IGdlbmVyYXRvci5hcHBseSh0aGlzQXJnLCBfYXJndW1lbnRzIHx8IFtdKSwgaSwgcSA9IFtdO1xyXG4gICAgcmV0dXJuIGkgPSB7fSwgdmVyYihcIm5leHRcIiksIHZlcmIoXCJ0aHJvd1wiKSwgdmVyYihcInJldHVyblwiKSwgaVtTeW1ib2wuYXN5bmNJdGVyYXRvcl0gPSBmdW5jdGlvbiAoKSB7IHJldHVybiB0aGlzOyB9LCBpO1xyXG4gICAgZnVuY3Rpb24gdmVyYihuKSB7IGlmIChnW25dKSBpW25dID0gZnVuY3Rpb24gKHYpIHsgcmV0dXJuIG5ldyBQcm9taXNlKGZ1bmN0aW9uIChhLCBiKSB7IHEucHVzaChbbiwgdiwgYSwgYl0pID4gMSB8fCByZXN1bWUobiwgdik7IH0pOyB9OyB9XHJcbiAgICBmdW5jdGlvbiByZXN1bWUobiwgdikgeyB0cnkgeyBzdGVwKGdbbl0odikpOyB9IGNhdGNoIChlKSB7IHNldHRsZShxWzBdWzNdLCBlKTsgfSB9XHJcbiAgICBmdW5jdGlvbiBzdGVwKHIpIHsgci52YWx1ZSBpbnN0YW5jZW9mIF9fYXdhaXQgPyBQcm9taXNlLnJlc29sdmUoci52YWx1ZS52KS50aGVuKGZ1bGZpbGwsIHJlamVjdCkgOiBzZXR0bGUocVswXVsyXSwgcik7IH1cclxuICAgIGZ1bmN0aW9uIGZ1bGZpbGwodmFsdWUpIHsgcmVzdW1lKFwibmV4dFwiLCB2YWx1ZSk7IH1cclxuICAgIGZ1bmN0aW9uIHJlamVjdCh2YWx1ZSkgeyByZXN1bWUoXCJ0aHJvd1wiLCB2YWx1ZSk7IH1cclxuICAgIGZ1bmN0aW9uIHNldHRsZShmLCB2KSB7IGlmIChmKHYpLCBxLnNoaWZ0KCksIHEubGVuZ3RoKSByZXN1bWUocVswXVswXSwgcVswXVsxXSk7IH1cclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fYXN5bmNEZWxlZ2F0b3Iobykge1xyXG4gICAgdmFyIGksIHA7XHJcbiAgICByZXR1cm4gaSA9IHt9LCB2ZXJiKFwibmV4dFwiKSwgdmVyYihcInRocm93XCIsIGZ1bmN0aW9uIChlKSB7IHRocm93IGU7IH0pLCB2ZXJiKFwicmV0dXJuXCIpLCBpW1N5bWJvbC5pdGVyYXRvcl0gPSBmdW5jdGlvbiAoKSB7IHJldHVybiB0aGlzOyB9LCBpO1xyXG4gICAgZnVuY3Rpb24gdmVyYihuLCBmKSB7IGlbbl0gPSBvW25dID8gZnVuY3Rpb24gKHYpIHsgcmV0dXJuIChwID0gIXApID8geyB2YWx1ZTogX19hd2FpdChvW25dKHYpKSwgZG9uZTogbiA9PT0gXCJyZXR1cm5cIiB9IDogZiA/IGYodikgOiB2OyB9IDogZjsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hc3luY1ZhbHVlcyhvKSB7XHJcbiAgICBpZiAoIVN5bWJvbC5hc3luY0l0ZXJhdG9yKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiU3ltYm9sLmFzeW5jSXRlcmF0b3IgaXMgbm90IGRlZmluZWQuXCIpO1xyXG4gICAgdmFyIG0gPSBvW1N5bWJvbC5hc3luY0l0ZXJhdG9yXSwgaTtcclxuICAgIHJldHVybiBtID8gbS5jYWxsKG8pIDogKG8gPSB0eXBlb2YgX192YWx1ZXMgPT09IFwiZnVuY3Rpb25cIiA/IF9fdmFsdWVzKG8pIDogb1tTeW1ib2wuaXRlcmF0b3JdKCksIGkgPSB7fSwgdmVyYihcIm5leHRcIiksIHZlcmIoXCJ0aHJvd1wiKSwgdmVyYihcInJldHVyblwiKSwgaVtTeW1ib2wuYXN5bmNJdGVyYXRvcl0gPSBmdW5jdGlvbiAoKSB7IHJldHVybiB0aGlzOyB9LCBpKTtcclxuICAgIGZ1bmN0aW9uIHZlcmIobikgeyBpW25dID0gb1tuXSAmJiBmdW5jdGlvbiAodikgeyByZXR1cm4gbmV3IFByb21pc2UoZnVuY3Rpb24gKHJlc29sdmUsIHJlamVjdCkgeyB2ID0gb1tuXSh2KSwgc2V0dGxlKHJlc29sdmUsIHJlamVjdCwgdi5kb25lLCB2LnZhbHVlKTsgfSk7IH07IH1cclxuICAgIGZ1bmN0aW9uIHNldHRsZShyZXNvbHZlLCByZWplY3QsIGQsIHYpIHsgUHJvbWlzZS5yZXNvbHZlKHYpLnRoZW4oZnVuY3Rpb24odikgeyByZXNvbHZlKHsgdmFsdWU6IHYsIGRvbmU6IGQgfSk7IH0sIHJlamVjdCk7IH1cclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fbWFrZVRlbXBsYXRlT2JqZWN0KGNvb2tlZCwgcmF3KSB7XHJcbiAgICBpZiAoT2JqZWN0LmRlZmluZVByb3BlcnR5KSB7IE9iamVjdC5kZWZpbmVQcm9wZXJ0eShjb29rZWQsIFwicmF3XCIsIHsgdmFsdWU6IHJhdyB9KTsgfSBlbHNlIHsgY29va2VkLnJhdyA9IHJhdzsgfVxyXG4gICAgcmV0dXJuIGNvb2tlZDtcclxufTtcclxuXHJcbnZhciBfX3NldE1vZHVsZURlZmF1bHQgPSBPYmplY3QuY3JlYXRlID8gKGZ1bmN0aW9uKG8sIHYpIHtcclxuICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShvLCBcImRlZmF1bHRcIiwgeyBlbnVtZXJhYmxlOiB0cnVlLCB2YWx1ZTogdiB9KTtcclxufSkgOiBmdW5jdGlvbihvLCB2KSB7XHJcbiAgICBvW1wiZGVmYXVsdFwiXSA9IHY7XHJcbn07XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19pbXBvcnRTdGFyKG1vZCkge1xyXG4gICAgaWYgKG1vZCAmJiBtb2QuX19lc01vZHVsZSkgcmV0dXJuIG1vZDtcclxuICAgIHZhciByZXN1bHQgPSB7fTtcclxuICAgIGlmIChtb2QgIT0gbnVsbCkgZm9yICh2YXIgayBpbiBtb2QpIGlmIChrICE9PSBcImRlZmF1bHRcIiAmJiBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwobW9kLCBrKSkgX19jcmVhdGVCaW5kaW5nKHJlc3VsdCwgbW9kLCBrKTtcclxuICAgIF9fc2V0TW9kdWxlRGVmYXVsdChyZXN1bHQsIG1vZCk7XHJcbiAgICByZXR1cm4gcmVzdWx0O1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19pbXBvcnREZWZhdWx0KG1vZCkge1xyXG4gICAgcmV0dXJuIChtb2QgJiYgbW9kLl9fZXNNb2R1bGUpID8gbW9kIDogeyBkZWZhdWx0OiBtb2QgfTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fY2xhc3NQcml2YXRlRmllbGRHZXQocmVjZWl2ZXIsIHN0YXRlLCBraW5kLCBmKSB7XHJcbiAgICBpZiAoa2luZCA9PT0gXCJhXCIgJiYgIWYpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJQcml2YXRlIGFjY2Vzc29yIHdhcyBkZWZpbmVkIHdpdGhvdXQgYSBnZXR0ZXJcIik7XHJcbiAgICBpZiAodHlwZW9mIHN0YXRlID09PSBcImZ1bmN0aW9uXCIgPyByZWNlaXZlciAhPT0gc3RhdGUgfHwgIWYgOiAhc3RhdGUuaGFzKHJlY2VpdmVyKSkgdGhyb3cgbmV3IFR5cGVFcnJvcihcIkNhbm5vdCByZWFkIHByaXZhdGUgbWVtYmVyIGZyb20gYW4gb2JqZWN0IHdob3NlIGNsYXNzIGRpZCBub3QgZGVjbGFyZSBpdFwiKTtcclxuICAgIHJldHVybiBraW5kID09PSBcIm1cIiA/IGYgOiBraW5kID09PSBcImFcIiA/IGYuY2FsbChyZWNlaXZlcikgOiBmID8gZi52YWx1ZSA6IHN0YXRlLmdldChyZWNlaXZlcik7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2NsYXNzUHJpdmF0ZUZpZWxkU2V0KHJlY2VpdmVyLCBzdGF0ZSwgdmFsdWUsIGtpbmQsIGYpIHtcclxuICAgIGlmIChraW5kID09PSBcIm1cIikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlByaXZhdGUgbWV0aG9kIGlzIG5vdCB3cml0YWJsZVwiKTtcclxuICAgIGlmIChraW5kID09PSBcImFcIiAmJiAhZikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlByaXZhdGUgYWNjZXNzb3Igd2FzIGRlZmluZWQgd2l0aG91dCBhIHNldHRlclwiKTtcclxuICAgIGlmICh0eXBlb2Ygc3RhdGUgPT09IFwiZnVuY3Rpb25cIiA/IHJlY2VpdmVyICE9PSBzdGF0ZSB8fCAhZiA6ICFzdGF0ZS5oYXMocmVjZWl2ZXIpKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiQ2Fubm90IHdyaXRlIHByaXZhdGUgbWVtYmVyIHRvIGFuIG9iamVjdCB3aG9zZSBjbGFzcyBkaWQgbm90IGRlY2xhcmUgaXRcIik7XHJcbiAgICByZXR1cm4gKGtpbmQgPT09IFwiYVwiID8gZi5jYWxsKHJlY2VpdmVyLCB2YWx1ZSkgOiBmID8gZi52YWx1ZSA9IHZhbHVlIDogc3RhdGUuc2V0KHJlY2VpdmVyLCB2YWx1ZSkpLCB2YWx1ZTtcclxufVxyXG4iLCJpbXBvcnQgeyByZXNvbHZlLCBleHRuYW1lLCByZWxhdGl2ZSwgam9pbiwgcGFyc2UsIHBvc2l4IH0gZnJvbSBcInBhdGhcIjtcbmltcG9ydCB7IFJlYWRhYmxlIH0gZnJvbSBcInN0cmVhbVwiO1xuaW1wb3J0IHsgY2xpcGJvYXJkIH0gZnJvbSBcImVsZWN0cm9uXCI7XG5cbmV4cG9ydCBmdW5jdGlvbiBpc0FuSW1hZ2UoZXh0OiBzdHJpbmcpIHtcbiAgcmV0dXJuIFtcIi5wbmdcIiwgXCIuanBnXCIsIFwiLmpwZWdcIiwgXCIuYm1wXCIsIFwiLmdpZlwiLCBcIi5zdmdcIiwgXCIudGlmZlwiXS5pbmNsdWRlcyhcbiAgICBleHQudG9Mb3dlckNhc2UoKVxuICApO1xufVxuZXhwb3J0IGZ1bmN0aW9uIGlzQXNzZXRUeXBlQW5JbWFnZShwYXRoOiBzdHJpbmcpOiBCb29sZWFuIHtcbiAgcmV0dXJuIChcbiAgICBbXCIucG5nXCIsIFwiLmpwZ1wiLCBcIi5qcGVnXCIsIFwiLmJtcFwiLCBcIi5naWZcIiwgXCIuc3ZnXCIsIFwiLnRpZmZcIl0uaW5kZXhPZihcbiAgICAgIGV4dG5hbWUocGF0aCkudG9Mb3dlckNhc2UoKVxuICAgICkgIT09IC0xXG4gICk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRPUygpIHtcbiAgY29uc3QgeyBhcHBWZXJzaW9uIH0gPSBuYXZpZ2F0b3I7XG4gIGlmIChhcHBWZXJzaW9uLmluZGV4T2YoXCJXaW5cIikgIT09IC0xKSB7XG4gICAgcmV0dXJuIFwiV2luZG93c1wiO1xuICB9IGVsc2UgaWYgKGFwcFZlcnNpb24uaW5kZXhPZihcIk1hY1wiKSAhPT0gLTEpIHtcbiAgICByZXR1cm4gXCJNYWNPU1wiO1xuICB9IGVsc2UgaWYgKGFwcFZlcnNpb24uaW5kZXhPZihcIlgxMVwiKSAhPT0gLTEpIHtcbiAgICByZXR1cm4gXCJMaW51eFwiO1xuICB9IGVsc2Uge1xuICAgIHJldHVybiBcIlVua25vd24gT1NcIjtcbiAgfVxufVxuZXhwb3J0IGFzeW5jIGZ1bmN0aW9uIHN0cmVhbVRvU3RyaW5nKHN0cmVhbTogUmVhZGFibGUpIHtcbiAgY29uc3QgY2h1bmtzID0gW107XG5cbiAgZm9yIGF3YWl0IChjb25zdCBjaHVuayBvZiBzdHJlYW0pIHtcbiAgICBjaHVua3MucHVzaChCdWZmZXIuZnJvbShjaHVuaykpO1xuICB9XG5cbiAgcmV0dXJuIEJ1ZmZlci5jb25jYXQoY2h1bmtzKS50b1N0cmluZyhcInV0Zi04XCIpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0VXJsQXNzZXQodXJsOiBzdHJpbmcpIHtcbiAgcmV0dXJuICh1cmwgPSB1cmwuc3Vic3RyKDEgKyB1cmwubGFzdEluZGV4T2YoXCIvXCIpKS5zcGxpdChcIj9cIilbMF0pLnNwbGl0KFxuICAgIFwiI1wiXG4gIClbMF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc0NvcHlJbWFnZUZpbGUoKSB7XG4gIGxldCBmaWxlUGF0aCA9IFwiXCI7XG4gIGNvbnN0IG9zID0gZ2V0T1MoKTtcblxuICBpZiAob3MgPT09IFwiV2luZG93c1wiKSB7XG4gICAgdmFyIHJhd0ZpbGVQYXRoID0gY2xpcGJvYXJkLnJlYWQoXCJGaWxlTmFtZVdcIik7XG4gICAgZmlsZVBhdGggPSByYXdGaWxlUGF0aC5yZXBsYWNlKG5ldyBSZWdFeHAoU3RyaW5nLmZyb21DaGFyQ29kZSgwKSwgXCJnXCIpLCBcIlwiKTtcbiAgfSBlbHNlIGlmIChvcyA9PT0gXCJNYWNPU1wiKSB7XG4gICAgZmlsZVBhdGggPSBjbGlwYm9hcmQucmVhZChcInB1YmxpYy5maWxlLXVybFwiKS5yZXBsYWNlKFwiZmlsZTovL1wiLCBcIlwiKTtcbiAgfSBlbHNlIHtcbiAgICBmaWxlUGF0aCA9IFwiXCI7XG4gIH1cbiAgcmV0dXJuIGlzQXNzZXRUeXBlQW5JbWFnZShmaWxlUGF0aCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRMYXN0SW1hZ2UobGlzdDogc3RyaW5nW10pIHtcbiAgY29uc3QgcmV2ZXJzZWRMaXN0ID0gbGlzdC5yZXZlcnNlKCk7XG4gIGxldCBsYXN0SW1hZ2U7XG4gIHJldmVyc2VkTGlzdC5mb3JFYWNoKGl0ZW0gPT4ge1xuICAgIGlmIChpdGVtICYmIGl0ZW0uc3RhcnRzV2l0aChcImh0dHBcIikpIHtcbiAgICAgIGxhc3RJbWFnZSA9IGl0ZW07XG4gICAgICByZXR1cm4gaXRlbTtcbiAgICB9XG4gIH0pO1xuICByZXR1cm4gbGFzdEltYWdlO1xufVxuIiwiaW1wb3J0IHsgUGx1Z2luU2V0dGluZ3MgfSBmcm9tIFwiLi9zZXR0aW5nXCI7XG5pbXBvcnQgeyBzdHJlYW1Ub1N0cmluZywgZ2V0TGFzdEltYWdlIH0gZnJvbSBcIi4vdXRpbHNcIjtcbmltcG9ydCB7IGV4ZWMgfSBmcm9tIFwiY2hpbGRfcHJvY2Vzc1wiO1xuaW1wb3J0IHsgTm90aWNlLCByZXF1ZXN0VXJsIH0gZnJvbSBcIm9ic2lkaWFuXCI7XG5cbmludGVyZmFjZSBQaWNHb1Jlc3BvbnNlIHtcbiAgc3VjY2Vzczogc3RyaW5nO1xuICBtc2c6IHN0cmluZztcbiAgcmVzdWx0OiBzdHJpbmdbXTtcbn1cblxuZXhwb3J0IGNsYXNzIFBpY0dvVXBsb2FkZXIge1xuICBzZXR0aW5nczogUGx1Z2luU2V0dGluZ3M7XG5cbiAgY29uc3RydWN0b3Ioc2V0dGluZ3M6IFBsdWdpblNldHRpbmdzKSB7XG4gICAgdGhpcy5zZXR0aW5ncyA9IHNldHRpbmdzO1xuICB9XG5cbiAgYXN5bmMgdXBsb2FkRmlsZXMoZmlsZUxpc3Q6IEFycmF5PFN0cmluZz4pOiBQcm9taXNlPGFueT4ge1xuICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgcmVxdWVzdFVybCh7XG4gICAgICB1cmw6IHRoaXMuc2V0dGluZ3MudXBsb2FkU2VydmVyLFxuICAgICAgbWV0aG9kOiBcIlBPU1RcIixcbiAgICAgIGhlYWRlcnM6IHsgXCJDb250ZW50LVR5cGVcIjogXCJhcHBsaWNhdGlvbi9qc29uXCIgfSxcbiAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHsgbGlzdDogZmlsZUxpc3QgfSksXG4gICAgfSk7XG5cbiAgICBjb25zdCBkYXRhID0gcmVzcG9uc2UuanNvbjtcbiAgICByZXR1cm4gZGF0YTtcbiAgfVxuXG4gIGFzeW5jIHVwbG9hZEZpbGVCeUNsaXBib2FyZCgpOiBQcm9taXNlPGFueT4ge1xuICAgIGNvbnN0IHJlcyA9IGF3YWl0IHJlcXVlc3RVcmwoe1xuICAgICAgdXJsOiB0aGlzLnNldHRpbmdzLnVwbG9hZFNlcnZlcixcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgfSk7XG5cbiAgICBsZXQgZGF0YTogUGljR29SZXNwb25zZSA9IHJlcy5qc29uO1xuXG4gICAgaWYgKHJlcy5zdGF0dXMgIT09IDIwMCkge1xuICAgICAgbGV0IGVyciA9IHsgcmVzcG9uc2U6IGRhdGEsIGJvZHk6IGRhdGEubXNnIH07XG4gICAgICByZXR1cm4ge1xuICAgICAgICBjb2RlOiAtMSxcbiAgICAgICAgbXNnOiBkYXRhLm1zZyxcbiAgICAgICAgZGF0YTogXCJcIixcbiAgICAgIH07XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiB7XG4gICAgICAgIGNvZGU6IDAsXG4gICAgICAgIG1zZzogXCJzdWNjZXNzXCIsXG4gICAgICAgIGRhdGE6IGRhdGEucmVzdWx0WzBdLFxuICAgICAgfTtcbiAgICB9XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIFBpY0dvQ29yZVVwbG9hZGVyIHtcbiAgc2V0dGluZ3M6IFBsdWdpblNldHRpbmdzO1xuXG4gIGNvbnN0cnVjdG9yKHNldHRpbmdzOiBQbHVnaW5TZXR0aW5ncykge1xuICAgIHRoaXMuc2V0dGluZ3MgPSBzZXR0aW5ncztcbiAgfVxuXG4gIGFzeW5jIHVwbG9hZEZpbGVzKGZpbGVMaXN0OiBBcnJheTxTdHJpbmc+KTogUHJvbWlzZTxhbnk+IHtcbiAgICBjb25zdCBsZW5ndGggPSBmaWxlTGlzdC5sZW5ndGg7XG4gICAgbGV0IGNsaSA9IHRoaXMuc2V0dGluZ3MucGljZ29Db3JlUGF0aCB8fCBcInBpY2dvXCI7XG4gICAgbGV0IGNvbW1hbmQgPSBgJHtjbGl9IHVwbG9hZCAke2ZpbGVMaXN0XG4gICAgICAubWFwKGl0ZW0gPT4gYFwiJHtpdGVtfVwiYClcbiAgICAgIC5qb2luKFwiIFwiKX1gO1xuXG4gICAgY29uc3QgcmVzID0gYXdhaXQgdGhpcy5leGVjKGNvbW1hbmQpO1xuICAgIGNvbnN0IHNwbGl0TGlzdCA9IHJlcy5zcGxpdChcIlxcblwiKTtcbiAgICBjb25zdCBzcGxpdExpc3RMZW5ndGggPSBzcGxpdExpc3QubGVuZ3RoO1xuICAgIGNvbnNvbGUubG9nKHNwbGl0TGlzdExlbmd0aCk7XG5cbiAgICBjb25zdCBkYXRhID0gc3BsaXRMaXN0LnNwbGljZShzcGxpdExpc3RMZW5ndGggLSAxIC0gbGVuZ3RoLCBsZW5ndGgpO1xuXG4gICAgaWYgKHJlcy5pbmNsdWRlcyhcIlBpY0dvIEVSUk9SXCIpKSB7XG4gICAgICByZXR1cm4ge1xuICAgICAgICBzdWNjZXNzOiBmYWxzZSxcbiAgICAgICAgbXNnOiBcIuWksei0pVwiLFxuICAgICAgfTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgc3VjY2VzczogdHJ1ZSxcbiAgICAgICAgcmVzdWx0OiBkYXRhLFxuICAgICAgfTtcbiAgICB9XG4gICAgLy8ge3N1Y2Nlc3M6dHJ1ZSxyZXN1bHQ6W119XG4gIH1cblxuICAvLyBQaWNHby1Db3JlIOS4iuS8oOWkhOeQhlxuICBhc3luYyB1cGxvYWRGaWxlQnlDbGlwYm9hcmQoKSB7XG4gICAgY29uc3QgcmVzID0gYXdhaXQgdGhpcy51cGxvYWRCeUNsaXAoKTtcbiAgICBjb25zdCBzcGxpdExpc3QgPSByZXMuc3BsaXQoXCJcXG5cIik7XG4gICAgY29uc3QgbGFzdEltYWdlID0gZ2V0TGFzdEltYWdlKHNwbGl0TGlzdCk7XG5cbiAgICBpZiAobGFzdEltYWdlKSB7XG4gICAgICByZXR1cm4ge1xuICAgICAgICBjb2RlOiAwLFxuICAgICAgICBtc2c6IFwic3VjY2Vzc1wiLFxuICAgICAgICBkYXRhOiBsYXN0SW1hZ2UsXG4gICAgICB9O1xuICAgIH0gZWxzZSB7XG4gICAgICBuZXcgTm90aWNlKGBcIlBsZWFzZSBjaGVjayBQaWNHby1Db3JlIGNvbmZpZ1wiXFxuJHtyZXN9YCk7XG4gICAgICByZXR1cm4ge1xuICAgICAgICBjb2RlOiAtMSxcbiAgICAgICAgbXNnOiBgXCJQbGVhc2UgY2hlY2sgUGljR28tQ29yZSBjb25maWdcIlxcbiR7cmVzfWAsXG4gICAgICAgIGRhdGE6IFwiXCIsXG4gICAgICB9O1xuICAgIH1cbiAgfVxuXG4gIC8vIFBpY0dvLUNvcmXnmoTliarliIfkuIrkvKDlj43ppohcbiAgYXN5bmMgdXBsb2FkQnlDbGlwKCkge1xuICAgIGxldCBjb21tYW5kO1xuICAgIGlmICh0aGlzLnNldHRpbmdzLnBpY2dvQ29yZVBhdGgpIHtcbiAgICAgIGNvbW1hbmQgPSBgJHt0aGlzLnNldHRpbmdzLnBpY2dvQ29yZVBhdGh9IHVwbG9hZGA7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbW1hbmQgPSBgcGljZ28gdXBsb2FkYDtcbiAgICB9XG4gICAgY29uc3QgcmVzID0gYXdhaXQgdGhpcy5leGVjKGNvbW1hbmQpO1xuICAgIHJldHVybiByZXM7XG4gIH1cbiAgYXN5bmMgZXhlYyhjb21tYW5kOiBzdHJpbmcpIHtcbiAgICBsZXQgeyBzdGRvdXQgfSA9IGF3YWl0IGV4ZWMoY29tbWFuZCk7XG4gICAgY29uc3QgcmVzID0gYXdhaXQgc3RyZWFtVG9TdHJpbmcoc3Rkb3V0KTtcbiAgICByZXR1cm4gcmVzO1xuICB9XG59XG4iLCJpbXBvcnQge1xuICBNYXJrZG93blZpZXcsXG4gIFBsdWdpbixcbiAgRmlsZVN5c3RlbUFkYXB0ZXIsXG4gIEVkaXRvcixcbiAgTWVudSxcbiAgTWVudUl0ZW0sXG4gIFRGaWxlLFxuICBub3JtYWxpemVQYXRoLFxuICBOb3RpY2UsXG4gIGFkZEljb24sXG4gIEFwcCxcbn0gZnJvbSBcIm9ic2lkaWFuXCI7XG5pbXBvcnQgeyBwYXJzZSB9IGZyb20gXCJwYXRoXCI7XG5cbmludGVyZmFjZSBJbWFnZSB7XG4gIHBhdGg6IHN0cmluZztcbiAgbmFtZTogc3RyaW5nO1xuICBzb3VyY2U6IHN0cmluZztcbn1cbmNvbnN0IFJFR0VYX0ZJTEUgPSAvXFwhXFxbKC4qPylcXF1cXCgoLio/KVxcKS9nO1xuY29uc3QgUkVHRVhfV0lLSV9GSUxFID0gL1xcIVxcW1xcWyguKj8pXFxdXFxdL2c7XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIEhlbHBlciB7XG4gIGFwcDogQXBwO1xuXG4gIGNvbnN0cnVjdG9yKGFwcDogQXBwKSB7XG4gICAgdGhpcy5hcHAgPSBhcHA7XG4gIH1cbiAgZ2V0RnJvbnRtYXR0ZXJWYWx1ZShrZXk6IHN0cmluZywgZGVmYXVsdFZhbHVlOiBhbnkgPSB1bmRlZmluZWQpIHtcbiAgICBjb25zdCBmaWxlID0gdGhpcy5hcHAud29ya3NwYWNlLmdldEFjdGl2ZUZpbGUoKTtcbiAgICBpZiAoIWZpbGUpIHtcbiAgICAgIHJldHVybiB1bmRlZmluZWQ7XG4gICAgfVxuICAgIGNvbnN0IHBhdGggPSBmaWxlLnBhdGg7XG4gICAgY29uc3QgY2FjaGUgPSB0aGlzLmFwcC5tZXRhZGF0YUNhY2hlLmdldENhY2hlKHBhdGgpO1xuXG4gICAgbGV0IHZhbHVlID0gZGVmYXVsdFZhbHVlO1xuICAgIGlmIChjYWNoZT8uZnJvbnRtYXR0ZXIgJiYgY2FjaGUuZnJvbnRtYXR0ZXIuaGFzT3duUHJvcGVydHkoa2V5KSkge1xuICAgICAgdmFsdWUgPSBjYWNoZS5mcm9udG1hdHRlcltrZXldO1xuICAgIH1cbiAgICByZXR1cm4gdmFsdWU7XG4gIH1cblxuICBnZXRFZGl0b3IoKSB7XG4gICAgY29uc3QgbWRWaWV3ID0gdGhpcy5hcHAud29ya3NwYWNlLmdldEFjdGl2ZVZpZXdPZlR5cGUoTWFya2Rvd25WaWV3KTtcbiAgICBpZiAobWRWaWV3KSB7XG4gICAgICByZXR1cm4gbWRWaWV3LmVkaXRvcjtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICB9XG4gIGdldFZhbHVlKCkge1xuICAgIGNvbnN0IGVkaXRvciA9IHRoaXMuZ2V0RWRpdG9yKCk7XG4gICAgcmV0dXJuIGVkaXRvci5nZXRWYWx1ZSgpO1xuICB9XG5cbiAgc2V0VmFsdWUodmFsdWU6IHN0cmluZykge1xuICAgIGNvbnN0IGVkaXRvciA9IHRoaXMuZ2V0RWRpdG9yKCk7XG4gICAgY29uc3QgeyBsZWZ0LCB0b3AgfSA9IGVkaXRvci5nZXRTY3JvbGxJbmZvKCk7XG4gICAgY29uc3QgcG9zaXRpb24gPSBlZGl0b3IuZ2V0Q3Vyc29yKCk7XG5cbiAgICBlZGl0b3Iuc2V0VmFsdWUodmFsdWUpO1xuICAgIGVkaXRvci5zY3JvbGxUbyhsZWZ0LCB0b3ApO1xuICAgIGVkaXRvci5zZXRDdXJzb3IocG9zaXRpb24pO1xuICB9XG5cbiAgLy8gZ2V0IGFsbCBmaWxlIHVybHMsIGluY2x1ZGUgbG9jYWwgYW5kIGludGVybmV0XG4gIGdldEFsbEZpbGVzKCk6IEltYWdlW10ge1xuICAgIGNvbnN0IGVkaXRvciA9IHRoaXMuZ2V0RWRpdG9yKCk7XG4gICAgbGV0IHZhbHVlID0gZWRpdG9yLmdldFZhbHVlKCk7XG4gICAgcmV0dXJuIHRoaXMuZ2V0SW1hZ2VMaW5rKHZhbHVlKTtcbiAgfVxuICBnZXRJbWFnZUxpbmsodmFsdWU6IHN0cmluZyk6IEltYWdlW10ge1xuICAgIGNvbnN0IG1hdGNoZXMgPSB2YWx1ZS5tYXRjaEFsbChSRUdFWF9GSUxFKTtcbiAgICBjb25zdCBXaWtpTWF0Y2hlcyA9IHZhbHVlLm1hdGNoQWxsKFJFR0VYX1dJS0lfRklMRSk7XG5cbiAgICBsZXQgZmlsZUFycmF5OiBJbWFnZVtdID0gW107XG5cbiAgICBmb3IgKGNvbnN0IG1hdGNoIG9mIG1hdGNoZXMpIHtcbiAgICAgIGNvbnN0IG5hbWUgPSBtYXRjaFsxXTtcbiAgICAgIGNvbnN0IHBhdGggPSBtYXRjaFsyXTtcbiAgICAgIGNvbnN0IHNvdXJjZSA9IG1hdGNoWzBdO1xuXG4gICAgICBmaWxlQXJyYXkucHVzaCh7XG4gICAgICAgIHBhdGg6IHBhdGgsXG4gICAgICAgIG5hbWU6IG5hbWUsXG4gICAgICAgIHNvdXJjZTogc291cmNlLFxuICAgICAgfSk7XG4gICAgfVxuXG4gICAgZm9yIChjb25zdCBtYXRjaCBvZiBXaWtpTWF0Y2hlcykge1xuICAgICAgY29uc29sZS5sb2cobWF0Y2gpO1xuXG4gICAgICBjb25zdCBuYW1lID0gcGFyc2UobWF0Y2hbMV0pLm5hbWU7XG4gICAgICBjb25zdCBwYXRoID0gbWF0Y2hbMV07XG4gICAgICBjb25zdCBzb3VyY2UgPSBtYXRjaFswXTtcblxuICAgICAgZmlsZUFycmF5LnB1c2goe1xuICAgICAgICBwYXRoOiBwYXRoLFxuICAgICAgICBuYW1lOiBuYW1lLFxuICAgICAgICBzb3VyY2U6IHNvdXJjZSxcbiAgICAgIH0pO1xuICAgIH1cbiAgICByZXR1cm4gZmlsZUFycmF5O1xuICB9XG59XG4iLCJpbXBvcnQgeyBBcHAsIFBsdWdpblNldHRpbmdUYWIsIFNldHRpbmcgfSBmcm9tIFwib2JzaWRpYW5cIjtcbmltcG9ydCBpbWFnZUF1dG9VcGxvYWRQbHVnaW4gZnJvbSBcIi4vbWFpblwiO1xuXG5leHBvcnQgaW50ZXJmYWNlIFBsdWdpblNldHRpbmdzIHtcbiAgdXBsb2FkQnlDbGlwU3dpdGNoOiBib29sZWFuO1xuICB1cGxvYWRTZXJ2ZXI6IHN0cmluZztcbiAgdXBsb2FkZXI6IHN0cmluZztcbiAgcGljZ29Db3JlUGF0aDogc3RyaW5nO1xuICBtZW51TW9kZTogc3RyaW5nO1xuICB3b3JrT25OZXRXb3JrOiBib29sZWFuO1xufVxuXG5leHBvcnQgY29uc3QgREVGQVVMVF9TRVRUSU5HUzogUGx1Z2luU2V0dGluZ3MgPSB7XG4gIHVwbG9hZEJ5Q2xpcFN3aXRjaDogdHJ1ZSxcbiAgdXBsb2FkZXI6IFwiUGljR29cIixcbiAgdXBsb2FkU2VydmVyOiBcImh0dHA6Ly8xMjcuMC4wLjE6MzY2NzcvdXBsb2FkXCIsXG4gIHBpY2dvQ29yZVBhdGg6IFwiXCIsXG4gIG1lbnVNb2RlOiBcImF1dG9cIixcbiAgd29ya09uTmV0V29yazogZmFsc2UsXG59O1xuXG5leHBvcnQgY2xhc3MgU2V0dGluZ1RhYiBleHRlbmRzIFBsdWdpblNldHRpbmdUYWIge1xuICBwbHVnaW46IGltYWdlQXV0b1VwbG9hZFBsdWdpbjtcblxuICBjb25zdHJ1Y3RvcihhcHA6IEFwcCwgcGx1Z2luOiBpbWFnZUF1dG9VcGxvYWRQbHVnaW4pIHtcbiAgICBzdXBlcihhcHAsIHBsdWdpbik7XG4gICAgdGhpcy5wbHVnaW4gPSBwbHVnaW47XG4gIH1cblxuICBkaXNwbGF5KCk6IHZvaWQge1xuICAgIGxldCB7IGNvbnRhaW5lckVsIH0gPSB0aGlzO1xuXG4gICAgY29udGFpbmVyRWwuZW1wdHkoKTtcbiAgICBjb250YWluZXJFbC5jcmVhdGVFbChcImgyXCIsIHsgdGV4dDogXCJwbHVnaW4gc2V0dGluZ3NcIiB9KTtcbiAgICBuZXcgU2V0dGluZyhjb250YWluZXJFbClcbiAgICAgIC5zZXROYW1lKFwiQXV0byBwYXN0ZWQgdXBsb2FkXCIpXG4gICAgICAuc2V0RGVzYyhcbiAgICAgICAgXCJpZiB5b3Ugc2V0IHRoaXMgdmFsdWUgdHJ1ZSwgd2hlbiB5b3UgcGFzdGUgaW1hZ2UsIGl0IHdpbGwgYmUgYXV0byB1cGxvYWRlZCh5b3Ugc2hvdWxkIHNldCB0aGUgcGljR28gc2VydmVyIHJpZ2h0bHkpXCJcbiAgICAgIClcbiAgICAgIC5hZGRUb2dnbGUodG9nZ2xlID0+XG4gICAgICAgIHRvZ2dsZVxuICAgICAgICAgIC5zZXRWYWx1ZSh0aGlzLnBsdWdpbi5zZXR0aW5ncy51cGxvYWRCeUNsaXBTd2l0Y2gpXG4gICAgICAgICAgLm9uQ2hhbmdlKGFzeW5jIHZhbHVlID0+IHtcbiAgICAgICAgICAgIHRoaXMucGx1Z2luLnNldHRpbmdzLnVwbG9hZEJ5Q2xpcFN3aXRjaCA9IHZhbHVlO1xuICAgICAgICAgICAgYXdhaXQgdGhpcy5wbHVnaW4uc2F2ZVNldHRpbmdzKCk7XG4gICAgICAgICAgfSlcbiAgICAgICk7XG5cbiAgICBuZXcgU2V0dGluZyhjb250YWluZXJFbClcbiAgICAgIC5zZXROYW1lKFwiRGVmYXVsdCB1cGxvYWRlclwiKVxuICAgICAgLnNldERlc2MoXCJEZWZhdWx0IHVwbG9hZGVyXCIpXG4gICAgICAuYWRkRHJvcGRvd24oY2IgPT5cbiAgICAgICAgY2JcbiAgICAgICAgICAuYWRkT3B0aW9uKFwiUGljR29cIiwgXCJQaWNHbyhhcHApXCIpXG4gICAgICAgICAgLmFkZE9wdGlvbihcIlBpY0dvLUNvcmVcIiwgXCJQaWNHby1Db3JlXCIpXG4gICAgICAgICAgLnNldFZhbHVlKHRoaXMucGx1Z2luLnNldHRpbmdzLnVwbG9hZGVyKVxuICAgICAgICAgIC5vbkNoYW5nZShhc3luYyB2YWx1ZSA9PiB7XG4gICAgICAgICAgICB0aGlzLnBsdWdpbi5zZXR0aW5ncy51cGxvYWRlciA9IHZhbHVlO1xuICAgICAgICAgICAgdGhpcy5kaXNwbGF5KCk7XG4gICAgICAgICAgICBhd2FpdCB0aGlzLnBsdWdpbi5zYXZlU2V0dGluZ3MoKTtcbiAgICAgICAgICB9KVxuICAgICAgKTtcblxuICAgIGlmICh0aGlzLnBsdWdpbi5zZXR0aW5ncy51cGxvYWRlciA9PT0gXCJQaWNHb1wiKSB7XG4gICAgICBuZXcgU2V0dGluZyhjb250YWluZXJFbClcbiAgICAgICAgLnNldE5hbWUoXCJQaWNHbyBzZXJ2ZXJcIilcbiAgICAgICAgLnNldERlc2MoXCJQaWNHbyBzZXJ2ZXJcIilcbiAgICAgICAgLmFkZFRleHQodGV4dCA9PlxuICAgICAgICAgIHRleHRcbiAgICAgICAgICAgIC5zZXRQbGFjZWhvbGRlcihcIlBsZWFzZSBpbnB1dCBQaWNHbyBzZXJ2ZXJcIilcbiAgICAgICAgICAgIC5zZXRWYWx1ZSh0aGlzLnBsdWdpbi5zZXR0aW5ncy51cGxvYWRTZXJ2ZXIpXG4gICAgICAgICAgICAub25DaGFuZ2UoYXN5bmMga2V5ID0+IHtcbiAgICAgICAgICAgICAgdGhpcy5wbHVnaW4uc2V0dGluZ3MudXBsb2FkU2VydmVyID0ga2V5O1xuICAgICAgICAgICAgICBhd2FpdCB0aGlzLnBsdWdpbi5zYXZlU2V0dGluZ3MoKTtcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgaWYgKHRoaXMucGx1Z2luLnNldHRpbmdzLnVwbG9hZGVyID09PSBcIlBpY0dvLUNvcmVcIikge1xuICAgICAgbmV3IFNldHRpbmcoY29udGFpbmVyRWwpXG4gICAgICAgIC5zZXROYW1lKFwiUGljR28tQ29yZSBwYXRoXCIpXG4gICAgICAgIC5zZXREZXNjKFxuICAgICAgICAgIFwiUGxlYXNlIGlucHV0IFBpY0dvLUNvcmUgcGF0aCwgZGVmYXVsdCB1c2luZyBlbnZpcm9ubWVudCB2YXJpYWJsZXNcIlxuICAgICAgICApXG4gICAgICAgIC5hZGRUZXh0KHRleHQgPT5cbiAgICAgICAgICB0ZXh0XG4gICAgICAgICAgICAuc2V0UGxhY2Vob2xkZXIoXCJcIilcbiAgICAgICAgICAgIC5zZXRWYWx1ZSh0aGlzLnBsdWdpbi5zZXR0aW5ncy5waWNnb0NvcmVQYXRoKVxuICAgICAgICAgICAgLm9uQ2hhbmdlKGFzeW5jIHZhbHVlID0+IHtcbiAgICAgICAgICAgICAgdGhpcy5wbHVnaW4uc2V0dGluZ3MucGljZ29Db3JlUGF0aCA9IHZhbHVlO1xuICAgICAgICAgICAgICBhd2FpdCB0aGlzLnBsdWdpbi5zYXZlU2V0dGluZ3MoKTtcbiAgICAgICAgICAgIH0pXG4gICAgICAgICk7XG4gICAgfVxuXG4gICAgbmV3IFNldHRpbmcoY29udGFpbmVyRWwpXG4gICAgICAuc2V0TmFtZShcIlVwbG9hZCBjb250ZXh0TWVudSBtb2RlXCIpXG4gICAgICAuc2V0RGVzYyhcbiAgICAgICAgXCJJdCBzaG91bGQgYmUgc2V0IGxpa2UgeW91ciBvYiBzZXR0aW5nLCBvdGhlcndpc2UgdGhlIGZlYXR1cmUgY2FuIG5vdCBiZSB3b3JrLlwiXG4gICAgICApXG4gICAgICAuYWRkRHJvcGRvd24oY2IgPT5cbiAgICAgICAgY2JcbiAgICAgICAgICAuYWRkT3B0aW9uKFwiYXV0b1wiLCBcImF1dG8oUmVhZCBmcm9tIGNvbmZpZylcIilcbiAgICAgICAgICAuYWRkT3B0aW9uKFwiYWJzb2x1dGVcIiwgXCJhYnNvbHV0ZVwiKVxuICAgICAgICAgIC5hZGRPcHRpb24oXCJyZWxhdGl2ZVwiLCBcInJlbGF0aXZlXCIpXG4gICAgICAgICAgLnNldFZhbHVlKHRoaXMucGx1Z2luLnNldHRpbmdzLm1lbnVNb2RlKVxuICAgICAgICAgIC5vbkNoYW5nZShhc3luYyB2YWx1ZSA9PiB7XG4gICAgICAgICAgICB0aGlzLnBsdWdpbi5zZXR0aW5ncy5tZW51TW9kZSA9IHZhbHVlO1xuICAgICAgICAgICAgdGhpcy5kaXNwbGF5KCk7XG4gICAgICAgICAgICBhd2FpdCB0aGlzLnBsdWdpbi5zYXZlU2V0dGluZ3MoKTtcbiAgICAgICAgICB9KVxuICAgICAgKTtcblxuICAgIG5ldyBTZXR0aW5nKGNvbnRhaW5lckVsKVxuICAgICAgLnNldE5hbWUoXCJXb3JrIG9uIG5ldHdvcmtcIilcbiAgICAgIC5zZXREZXNjKFxuICAgICAgICBcIldoZW4geW91IHBhc3RlLCBtZCBzdGFuZGFyZCBpbWFnZSBsaW5rIGluIHlvdXIgY2xpcGJvYXJkIHdpbGwgYmUgYXV0byB1cGxvYWQuXCJcbiAgICAgIClcbiAgICAgIC5hZGRUb2dnbGUodG9nZ2xlID0+XG4gICAgICAgIHRvZ2dsZVxuICAgICAgICAgIC5zZXRWYWx1ZSh0aGlzLnBsdWdpbi5zZXR0aW5ncy53b3JrT25OZXRXb3JrKVxuICAgICAgICAgIC5vbkNoYW5nZShhc3luYyB2YWx1ZSA9PiB7XG4gICAgICAgICAgICB0aGlzLnBsdWdpbi5zZXR0aW5ncy53b3JrT25OZXRXb3JrID0gdmFsdWU7XG4gICAgICAgICAgICB0aGlzLmRpc3BsYXkoKTtcbiAgICAgICAgICAgIGF3YWl0IHRoaXMucGx1Z2luLnNhdmVTZXR0aW5ncygpO1xuICAgICAgICAgIH0pXG4gICAgICApO1xuICB9XG59XG4iLCJpbXBvcnQge1xuICBNYXJrZG93blZpZXcsXG4gIFBsdWdpbixcbiAgRmlsZVN5c3RlbUFkYXB0ZXIsXG4gIEVkaXRvcixcbiAgTWVudSxcbiAgTWVudUl0ZW0sXG4gIFRGaWxlLFxuICBub3JtYWxpemVQYXRoLFxuICBOb3RpY2UsXG4gIGFkZEljb24sXG4gIHJlcXVlc3RVcmwsXG59IGZyb20gXCJvYnNpZGlhblwiO1xuXG5pbXBvcnQgeyByZXNvbHZlLCByZWxhdGl2ZSwgam9pbiwgcGFyc2UsIHBvc2l4IH0gZnJvbSBcInBhdGhcIjtcbmltcG9ydCB7IGV4aXN0c1N5bmMsIG1rZGlyU3luYywgd3JpdGVGaWxlU3luYyB9IGZyb20gXCJmc1wiO1xuXG5pbXBvcnQge1xuICBpc0Fzc2V0VHlwZUFuSW1hZ2UsXG4gIGlzQW5JbWFnZSxcbiAgZ2V0VXJsQXNzZXQsXG4gIGlzQ29weUltYWdlRmlsZSxcbn0gZnJvbSBcIi4vdXRpbHNcIjtcbmltcG9ydCB7IFBpY0dvVXBsb2FkZXIsIFBpY0dvQ29yZVVwbG9hZGVyIH0gZnJvbSBcIi4vdXBsb2FkZXJcIjtcbmltcG9ydCBIZWxwZXIgZnJvbSBcIi4vaGVscGVyXCI7XG5cbmltcG9ydCB7IFNldHRpbmdUYWIsIFBsdWdpblNldHRpbmdzLCBERUZBVUxUX1NFVFRJTkdTIH0gZnJvbSBcIi4vc2V0dGluZ1wiO1xuXG5pbnRlcmZhY2UgSW1hZ2Uge1xuICBwYXRoOiBzdHJpbmc7XG4gIG5hbWU6IHN0cmluZztcbiAgc291cmNlOiBzdHJpbmc7XG59XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGltYWdlQXV0b1VwbG9hZFBsdWdpbiBleHRlbmRzIFBsdWdpbiB7XG4gIHNldHRpbmdzOiBQbHVnaW5TZXR0aW5ncztcbiAgaGVscGVyOiBIZWxwZXI7XG4gIGVkaXRvcjogRWRpdG9yO1xuICBwaWNHb1VwbG9hZGVyOiBQaWNHb1VwbG9hZGVyO1xuICBwaWNHb0NvcmVVcGxvYWRlcjogUGljR29Db3JlVXBsb2FkZXI7XG4gIHVwbG9hZGVyOiBQaWNHb1VwbG9hZGVyIHwgUGljR29Db3JlVXBsb2FkZXI7XG5cbiAgYXN5bmMgbG9hZFNldHRpbmdzKCkge1xuICAgIHRoaXMuc2V0dGluZ3MgPSBPYmplY3QuYXNzaWduKERFRkFVTFRfU0VUVElOR1MsIGF3YWl0IHRoaXMubG9hZERhdGEoKSk7XG4gIH1cblxuICBhc3luYyBzYXZlU2V0dGluZ3MoKSB7XG4gICAgYXdhaXQgdGhpcy5zYXZlRGF0YSh0aGlzLnNldHRpbmdzKTtcbiAgfVxuXG4gIG9udW5sb2FkKCkge31cblxuICBhc3luYyBvbmxvYWQoKSB7XG4gICAgYXdhaXQgdGhpcy5sb2FkU2V0dGluZ3MoKTtcblxuICAgIHRoaXMuaGVscGVyID0gbmV3IEhlbHBlcih0aGlzLmFwcCk7XG4gICAgdGhpcy5waWNHb1VwbG9hZGVyID0gbmV3IFBpY0dvVXBsb2FkZXIodGhpcy5zZXR0aW5ncyk7XG4gICAgdGhpcy5waWNHb0NvcmVVcGxvYWRlciA9IG5ldyBQaWNHb0NvcmVVcGxvYWRlcih0aGlzLnNldHRpbmdzKTtcblxuICAgIGlmICh0aGlzLnNldHRpbmdzLnVwbG9hZGVyID09PSBcIlBpY0dvXCIpIHtcbiAgICAgIHRoaXMudXBsb2FkZXIgPSB0aGlzLnBpY0dvVXBsb2FkZXI7XG4gICAgfSBlbHNlIGlmICh0aGlzLnNldHRpbmdzLnVwbG9hZGVyID09PSBcIlBpY0dvLUNvcmVcIikge1xuICAgICAgdGhpcy51cGxvYWRlciA9IHRoaXMucGljR29Db3JlVXBsb2FkZXI7XG4gICAgfSBlbHNlIHtcbiAgICAgIG5ldyBOb3RpY2UoXCJ1bmtub3duIHVwbG9hZGVyXCIpO1xuICAgIH1cblxuICAgIGFkZEljb24oXG4gICAgICBcInVwbG9hZFwiLFxuICAgICAgYDxzdmcgdD1cIjE2MzY2MzA3ODM0MjlcIiBjbGFzcz1cImljb25cIiB2aWV3Qm94PVwiMCAwIDEwMCAxMDBcIiB2ZXJzaW9uPVwiMS4xXCIgcC1pZD1cIjQ2NDlcIiB4bWxucz1cImh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnXCI+XG4gICAgICA8cGF0aCBkPVwiTSA3MS42MzggMzUuMzM2IEwgNzkuNDA4IDM1LjMzNiBDIDgzLjcgMzUuMzM2IDg3LjE3OCAzOC42NjIgODcuMTc4IDQyLjc2NSBMIDg3LjE3OCA4NC44NjQgQyA4Ny4xNzggODguOTY5IDgzLjcgOTIuMjk1IDc5LjQwOCA5Mi4yOTUgTCAxNy4yNDkgOTIuMjk1IEMgMTIuOTU3IDkyLjI5NSA5LjQ3OSA4OC45NjkgOS40NzkgODQuODY0IEwgOS40NzkgNDIuNzY1IEMgOS40NzkgMzguNjYyIDEyLjk1NyAzNS4zMzYgMTcuMjQ5IDM1LjMzNiBMIDI1LjAxOSAzNS4zMzYgTCAyNS4wMTkgNDIuNzY1IEwgMTcuMjQ5IDQyLjc2NSBMIDE3LjI0OSA4NC44NjQgTCA3OS40MDggODQuODY0IEwgNzkuNDA4IDQyLjc2NSBMIDcxLjYzOCA0Mi43NjUgTCA3MS42MzggMzUuMzM2IFogTSA0OS4wMTQgMTAuMTc5IEwgNjcuMzI2IDI3LjY4OCBMIDYxLjgzNSAzMi45NDIgTCA1Mi44NDkgMjQuMzUyIEwgNTIuODQ5IDU5LjczMSBMIDQ1LjA3OCA1OS43MzEgTCA0NS4wNzggMjQuNDU1IEwgMzYuMTk0IDMyLjk0NyBMIDMwLjcwMiAyNy42OTIgTCA0OS4wMTIgMTAuMTgxIFpcIiBwLWlkPVwiNDY1MFwiIGZpbGw9XCIjOGE4YThhXCI+PC9wYXRoPlxuICAgIDwvc3ZnPmBcbiAgICApO1xuXG4gICAgdGhpcy5hZGRTZXR0aW5nVGFiKG5ldyBTZXR0aW5nVGFiKHRoaXMuYXBwLCB0aGlzKSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwidXBsb2FkIGFsbCBpbWFnZXNcIixcbiAgICAgIG5hbWU6IFwidXBsb2FkIGFsbCBpbWFnZXNcIixcbiAgICAgIGNoZWNrQ2FsbGJhY2s6IChjaGVja2luZzogYm9vbGVhbikgPT4ge1xuICAgICAgICBsZXQgbGVhZiA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgICAgICBpZiAobGVhZikge1xuICAgICAgICAgIGlmICghY2hlY2tpbmcpIHtcbiAgICAgICAgICAgIHRoaXMudXBsb2FkQWxsRmlsZSgpO1xuICAgICAgICAgIH1cbiAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICB9LFxuICAgIH0pO1xuICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJkb2FubG9hZCBhbGwgaW1hZ2VzXCIsXG4gICAgICBuYW1lOiBcImRvd25sb2FkIGFsbCBpbWFnZXNcIixcbiAgICAgIGNoZWNrQ2FsbGJhY2s6IChjaGVja2luZzogYm9vbGVhbikgPT4ge1xuICAgICAgICBsZXQgbGVhZiA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgICAgICBpZiAobGVhZikge1xuICAgICAgICAgIGlmICghY2hlY2tpbmcpIHtcbiAgICAgICAgICAgIHRoaXMuZG93bmxvYWRBbGxJbWFnZUZpbGVzKCk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH0sXG4gICAgfSk7XG5cbiAgICB0aGlzLnNldHVwUGFzdGVIYW5kbGVyKCk7XG4gICAgdGhpcy5yZWdpc3RlckZpbGVNZW51KCk7XG4gIH1cblxuICBhc3luYyBkb3dubG9hZEFsbEltYWdlRmlsZXMoKSB7XG4gICAgY29uc3QgZm9sZGVyUGF0aCA9IHRoaXMuZ2V0RmlsZUFzc2V0UGF0aCgpO1xuICAgIGNvbnN0IGZpbGVBcnJheSA9IHRoaXMuaGVscGVyLmdldEFsbEZpbGVzKCk7XG4gICAgaWYgKCFleGlzdHNTeW5jKGZvbGRlclBhdGgpKSB7XG4gICAgICBta2RpclN5bmMoZm9sZGVyUGF0aCk7XG4gICAgfVxuXG4gICAgbGV0IGltYWdlQXJyYXkgPSBbXTtcbiAgICBmb3IgKGNvbnN0IGZpbGUgb2YgZmlsZUFycmF5KSB7XG4gICAgICBpZiAoIWZpbGUucGF0aC5zdGFydHNXaXRoKFwiaHR0cFwiKSkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgdXJsID0gZmlsZS5wYXRoO1xuICAgICAgY29uc3QgYXNzZXQgPSBnZXRVcmxBc3NldCh1cmwpO1xuICAgICAgaWYgKCFpc0FuSW1hZ2UoYXNzZXQuc3Vic3RyKGFzc2V0Lmxhc3RJbmRleE9mKFwiLlwiKSkpKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgbGV0IFtuYW1lLCBleHRdID0gW1xuICAgICAgICBkZWNvZGVVUkkocGFyc2UoYXNzZXQpLm5hbWUpLnJlcGxhY2VBbGwoL1tcXFxcXFxcXC86Kj9cXFwiPD58XS9nLCBcIi1cIiksXG4gICAgICAgIHBhcnNlKGFzc2V0KS5leHQsXG4gICAgICBdO1xuICAgICAgLy8g5aaC5p6c5paH5Lu25ZCN5bey5a2Y5Zyo77yM5YiZ55So6ZqP5py65YC85pu/5o2iXG4gICAgICBpZiAoZXhpc3RzU3luYyhqb2luKGZvbGRlclBhdGgsIGVuY29kZVVSSShhc3NldCkpKSkge1xuICAgICAgICBuYW1lID0gKE1hdGgucmFuZG9tKCkgKyAxKS50b1N0cmluZygzNikuc3Vic3RyKDIsIDUpO1xuICAgICAgfVxuICAgICAgbmFtZSA9IGBpbWFnZS0ke25hbWV9YDtcblxuICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCB0aGlzLmRvd25sb2FkKFxuICAgICAgICB1cmwsXG4gICAgICAgIGpvaW4oZm9sZGVyUGF0aCwgYCR7bmFtZX0ke2V4dH1gKVxuICAgICAgKTtcbiAgICAgIGlmIChyZXNwb25zZS5vaykge1xuICAgICAgICBjb25zdCBhY3RpdmVGb2xkZXIgPSB0aGlzLmFwcC52YXVsdC5nZXRBYnN0cmFjdEZpbGVCeVBhdGgoXG4gICAgICAgICAgdGhpcy5hcHAud29ya3NwYWNlLmdldEFjdGl2ZUZpbGUoKS5wYXRoXG4gICAgICAgICkucGFyZW50LnBhdGg7XG5cbiAgICAgICAgY29uc3QgYmFzZVBhdGggPSAoXG4gICAgICAgICAgdGhpcy5hcHAudmF1bHQuYWRhcHRlciBhcyBGaWxlU3lzdGVtQWRhcHRlclxuICAgICAgICApLmdldEJhc2VQYXRoKCk7XG4gICAgICAgIGNvbnN0IGFic3RyYWN0QWN0aXZlRm9sZGVyID0gcmVzb2x2ZShiYXNlUGF0aCwgYWN0aXZlRm9sZGVyKTtcblxuICAgICAgICBpbWFnZUFycmF5LnB1c2goe1xuICAgICAgICAgIHNvdXJjZTogZmlsZS5zb3VyY2UsXG4gICAgICAgICAgbmFtZTogbmFtZSxcbiAgICAgICAgICBwYXRoOiBub3JtYWxpemVQYXRoKHJlbGF0aXZlKGFic3RyYWN0QWN0aXZlRm9sZGVyLCByZXNwb25zZS5wYXRoKSksXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIGxldCB2YWx1ZSA9IHRoaXMuaGVscGVyLmdldFZhbHVlKCk7XG4gICAgaW1hZ2VBcnJheS5tYXAoaW1hZ2UgPT4ge1xuICAgICAgdmFsdWUgPSB2YWx1ZS5yZXBsYWNlKFxuICAgICAgICBpbWFnZS5zb3VyY2UsXG4gICAgICAgIGAhWyR7aW1hZ2UubmFtZX1dKCR7ZW5jb2RlVVJJKGltYWdlLnBhdGgpfSlgXG4gICAgICApO1xuICAgIH0pO1xuXG4gICAgdGhpcy5oZWxwZXIuc2V0VmFsdWUodmFsdWUpO1xuXG4gICAgbmV3IE5vdGljZShcbiAgICAgIGBhbGw6ICR7ZmlsZUFycmF5Lmxlbmd0aH1cXG5zdWNjZXNzOiAke2ltYWdlQXJyYXkubGVuZ3RofVxcbmZhaWxlZDogJHtcbiAgICAgICAgZmlsZUFycmF5Lmxlbmd0aCAtIGltYWdlQXJyYXkubGVuZ3RoXG4gICAgICB9YFxuICAgICk7XG4gIH1cblxuICAvLyDojrflj5blvZPliY3mlofku7bmiYDlsZ7nmoTpmYTku7bmlofku7blpLlcbiAgZ2V0RmlsZUFzc2V0UGF0aCgpIHtcbiAgICBjb25zdCBiYXNlUGF0aCA9IChcbiAgICAgIHRoaXMuYXBwLnZhdWx0LmFkYXB0ZXIgYXMgRmlsZVN5c3RlbUFkYXB0ZXJcbiAgICApLmdldEJhc2VQYXRoKCk7XG5cbiAgICAvLyBAdHMtaWdub3JlXG4gICAgY29uc3QgYXNzZXRGb2xkZXI6IHN0cmluZyA9IHRoaXMuYXBwLnZhdWx0LmNvbmZpZy5hdHRhY2htZW50Rm9sZGVyUGF0aDtcbiAgICBjb25zdCBhY3RpdmVGaWxlID0gdGhpcy5hcHAudmF1bHQuZ2V0QWJzdHJhY3RGaWxlQnlQYXRoKFxuICAgICAgdGhpcy5hcHAud29ya3NwYWNlLmdldEFjdGl2ZUZpbGUoKS5wYXRoXG4gICAgKTtcblxuICAgIC8vIOW9k+WJjeaWh+S7tuWkueS4i+eahOWtkOaWh+S7tuWkuVxuICAgIGlmIChhc3NldEZvbGRlci5zdGFydHNXaXRoKFwiLi9cIikpIHtcbiAgICAgIGNvbnN0IGFjdGl2ZUZvbGRlciA9IGRlY29kZVVSSShyZXNvbHZlKGJhc2VQYXRoLCBhY3RpdmVGaWxlLnBhcmVudC5wYXRoKSk7XG4gICAgICByZXR1cm4gam9pbihhY3RpdmVGb2xkZXIsIGFzc2V0Rm9sZGVyKTtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8g5qC55paH5Lu25aS5XG4gICAgICByZXR1cm4gam9pbihiYXNlUGF0aCwgYXNzZXRGb2xkZXIpO1xuICAgIH1cbiAgfVxuXG4gIGFzeW5jIGRvd25sb2FkKHVybDogc3RyaW5nLCBwYXRoOiBzdHJpbmcpIHtcbiAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IHJlcXVlc3RVcmwoeyB1cmwgfSk7XG5cbiAgICBpZiAocmVzcG9uc2Uuc3RhdHVzICE9PSAyMDApIHtcbiAgICAgIHJldHVybiB7XG4gICAgICAgIG9rOiBmYWxzZSxcbiAgICAgICAgbXNnOiBcImVycm9yXCIsXG4gICAgICB9O1xuICAgIH1cbiAgICBjb25zdCBidWZmZXIgPSBCdWZmZXIuZnJvbShyZXNwb25zZS5hcnJheUJ1ZmZlcik7XG5cbiAgICB0cnkge1xuICAgICAgd3JpdGVGaWxlU3luYyhwYXRoLCBidWZmZXIpO1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgb2s6IHRydWUsXG4gICAgICAgIG1zZzogXCJva1wiLFxuICAgICAgICBwYXRoOiBwYXRoLFxuICAgICAgfTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUuZXJyb3IoZXJyKTtcblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgb2s6IGZhbHNlLFxuICAgICAgICBtc2c6IGVycixcbiAgICAgIH07XG4gICAgfVxuICB9XG5cbiAgcmVnaXN0ZXJGaWxlTWVudSgpIHtcbiAgICB0aGlzLnJlZ2lzdGVyRXZlbnQoXG4gICAgICB0aGlzLmFwcC53b3Jrc3BhY2Uub24oXG4gICAgICAgIFwiZmlsZS1tZW51XCIsXG4gICAgICAgIChtZW51OiBNZW51LCBmaWxlOiBURmlsZSwgc291cmNlOiBzdHJpbmcpID0+IHtcbiAgICAgICAgICBpZiAoIWlzQXNzZXRUeXBlQW5JbWFnZShmaWxlLnBhdGgpKSB7XG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgfVxuICAgICAgICAgIG1lbnUuYWRkSXRlbSgoaXRlbTogTWVudUl0ZW0pID0+IHtcbiAgICAgICAgICAgIGl0ZW1cbiAgICAgICAgICAgICAgLnNldFRpdGxlKFwiVXBsb2FkXCIpXG4gICAgICAgICAgICAgIC5zZXRJY29uKFwidXBsb2FkXCIpXG4gICAgICAgICAgICAgIC5vbkNsaWNrKCgpID0+IHtcbiAgICAgICAgICAgICAgICBpZiAoIShmaWxlIGluc3RhbmNlb2YgVEZpbGUpKSB7XG4gICAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgY29uc3QgYmFzZVBhdGggPSAoXG4gICAgICAgICAgICAgICAgICB0aGlzLmFwcC52YXVsdC5hZGFwdGVyIGFzIEZpbGVTeXN0ZW1BZGFwdGVyXG4gICAgICAgICAgICAgICAgKS5nZXRCYXNlUGF0aCgpO1xuXG4gICAgICAgICAgICAgICAgY29uc3QgdXJpID0gZGVjb2RlVVJJKHJlc29sdmUoYmFzZVBhdGgsIGZpbGUucGF0aCkpO1xuXG4gICAgICAgICAgICAgICAgdGhpcy51cGxvYWRlci51cGxvYWRGaWxlcyhbdXJpXSkudGhlbihyZXMgPT4ge1xuICAgICAgICAgICAgICAgICAgaWYgKHJlcy5zdWNjZXNzKSB7XG4gICAgICAgICAgICAgICAgICAgIC8vIEB0cy1pZ25vcmVcbiAgICAgICAgICAgICAgICAgICAgbGV0IHVwbG9hZFVybCA9IHJlcy5yZXN1bHRbMF07XG4gICAgICAgICAgICAgICAgICAgIGNvbnN0IHNvdXJjZVVyaSA9IGVuY29kZVVSSShcbiAgICAgICAgICAgICAgICAgICAgICByZWxhdGl2ZShcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuYXBwLndvcmtzcGFjZS5nZXRBY3RpdmVGaWxlKCkucGFyZW50LnBhdGgsXG4gICAgICAgICAgICAgICAgICAgICAgICBmaWxlLnBhdGhcbiAgICAgICAgICAgICAgICAgICAgICApLnJlcGxhY2VBbGwoXCJcXFxcXCIsIFwiL1wiKVxuICAgICAgICAgICAgICAgICAgICApO1xuXG4gICAgICAgICAgICAgICAgICAgIGxldCB2YWx1ZSA9IHRoaXMuaGVscGVyLmdldFZhbHVlKCk7XG4gICAgICAgICAgICAgICAgICAgIGxldCBtZW51TW9kZSA9IHRoaXMuc2V0dGluZ3MubWVudU1vZGU7XG4gICAgICAgICAgICAgICAgICAgIGlmIChtZW51TW9kZSA9PT0gXCJhdXRvXCIpIHtcbiAgICAgICAgICAgICAgICAgICAgICAvLyBAdHMtaWdub3JlXG4gICAgICAgICAgICAgICAgICAgICAgbWVudU1vZGUgPSB0aGlzLmFwcC52YXVsdC5jb25maWcubmV3TGlua0Zvcm1hdDtcbiAgICAgICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgICAgIGlmIChtZW51TW9kZSA9PT0gXCJyZWxhdGl2ZVwiKSB7XG4gICAgICAgICAgICAgICAgICAgICAgLy8g5pu/5o2i55u45a+56Lev5b6E55qEICFbXSgp5qC85byPXG4gICAgICAgICAgICAgICAgICAgICAgdmFsdWUgPSB2YWx1ZS5yZXBsYWNlQWxsKHNvdXJjZVVyaSwgdXBsb2FkVXJsKTtcblxuICAgICAgICAgICAgICAgICAgICAgIC8vIOabv+aNouebuOWvuei3r+W+hOeahCAhW1tdXeagvOW8j1xuICAgICAgICAgICAgICAgICAgICAgIHZhbHVlID0gdmFsdWUucmVwbGFjZUFsbChcbiAgICAgICAgICAgICAgICAgICAgICAgIGAhW1ske2RlY29kZVVSSShzb3VyY2VVcmkpfV1dYCxcbiAgICAgICAgICAgICAgICAgICAgICAgIGAhW10oJHt1cGxvYWRVcmx9KWBcbiAgICAgICAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgICAgICB9IGVsc2UgaWYgKG1lbnVNb2RlID09PSBcImFic29sdXRlXCIpIHtcbiAgICAgICAgICAgICAgICAgICAgICAvLyDmm7/mjaLnu53lr7not6/lvoTnmoQgIVtbXV3moLzlvI9cbiAgICAgICAgICAgICAgICAgICAgICB2YWx1ZSA9IHZhbHVlLnJlcGxhY2VBbGwoXG4gICAgICAgICAgICAgICAgICAgICAgICBgIVtbJHtmaWxlLnBhdGh9XV1gLFxuICAgICAgICAgICAgICAgICAgICAgICAgYCFbXSgke3VwbG9hZFVybH0pYFxuICAgICAgICAgICAgICAgICAgICAgICk7XG5cbiAgICAgICAgICAgICAgICAgICAgICAvLyDmm7/mjaLnu53lr7not6/lvoTnmoQgIVtdKCnmoLzlvI9cbiAgICAgICAgICAgICAgICAgICAgICB2YWx1ZSA9IHZhbHVlLnJlcGxhY2VBbGwoXG4gICAgICAgICAgICAgICAgICAgICAgICBmaWxlLnBhdGgucmVwbGFjZUFsbChcIiBcIiwgXCIlMjBcIiksXG4gICAgICAgICAgICAgICAgICAgICAgICB1cGxvYWRVcmxcbiAgICAgICAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICAgIG5ldyBOb3RpY2UoYE5vdCBzdXBwb3J0ICR7bWVudU1vZGV9IG1vZGVgKTtcbiAgICAgICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuaGVscGVyLnNldFZhbHVlKHZhbHVlKTtcbiAgICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIG5ldyBOb3RpY2UocmVzLm1zZyB8fCBcIlVwbG9hZCBlcnJvclwiKTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgIClcbiAgICApO1xuICB9XG5cbiAgLy8gdXBsb2RhIGFsbCBmaWxlXG4gIHVwbG9hZEFsbEZpbGUoKSB7XG4gICAgbGV0IGtleSA9IHRoaXMuaGVscGVyLmdldFZhbHVlKCk7XG5cbiAgICBjb25zdCB0aGlzUGF0aCA9IHRoaXMuYXBwLnZhdWx0LmdldEFic3RyYWN0RmlsZUJ5UGF0aChcbiAgICAgIHRoaXMuYXBwLndvcmtzcGFjZS5nZXRBY3RpdmVGaWxlKCkucGF0aFxuICAgICk7XG4gICAgY29uc3QgYmFzZVBhdGggPSAoXG4gICAgICB0aGlzLmFwcC52YXVsdC5hZGFwdGVyIGFzIEZpbGVTeXN0ZW1BZGFwdGVyXG4gICAgKS5nZXRCYXNlUGF0aCgpO1xuXG4gICAgbGV0IGltYWdlTGlzdDogSW1hZ2VbXSA9IFtdO1xuICAgIGNvbnN0IGZpbGVBcnJheSA9IHRoaXMuaGVscGVyLmdldEFsbEZpbGVzKCk7XG5cbiAgICBmb3IgKGNvbnN0IG1hdGNoIG9mIGZpbGVBcnJheSkge1xuICAgICAgY29uc3QgaW1hZ2VOYW1lID0gbWF0Y2gubmFtZTtcbiAgICAgIGNvbnN0IGVuY29kZWRVcmkgPSBtYXRjaC5wYXRoO1xuICAgICAgaWYgKCFlbmNvZGVkVXJpLnN0YXJ0c1dpdGgoXCJodHRwXCIpKSB7XG4gICAgICAgIGxldCBhYnN0cmFjdEltYWdlRmlsZTtcbiAgICAgICAgLy8g5b2T6Lev5b6E5Lul4oCcLuKAneW8gOWktOaXtu+8jOivhuWIq+S4uuebuOWvuei3r+W+hO+8jOS4jeeEtuWwseiupOS4uuaXtue7neWvuei3r+W+hFxuICAgICAgICBpZiAoZW5jb2RlZFVyaS5zdGFydHNXaXRoKFwiLlwiKSkge1xuICAgICAgICAgIGFic3RyYWN0SW1hZ2VGaWxlID0gZGVjb2RlVVJJKFxuICAgICAgICAgICAgam9pbihcbiAgICAgICAgICAgICAgYmFzZVBhdGgsXG4gICAgICAgICAgICAgIHBvc2l4LnJlc29sdmUocG9zaXguam9pbihcIi9cIiwgdGhpc1BhdGgucGFyZW50LnBhdGgpLCBlbmNvZGVkVXJpKVxuICAgICAgICAgICAgKVxuICAgICAgICAgICk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgYWJzdHJhY3RJbWFnZUZpbGUgPSBkZWNvZGVVUkkoam9pbihiYXNlUGF0aCwgZW5jb2RlZFVyaSkpO1xuXG4gICAgICAgICAgLy8g5b2T6Kej5p6Q5Li657ud5a+56Lev5b6E5Y205om+5LiN5Yiw5paH5Lu277yM5bCd6K+V6Kej5p6Q5Li655u45a+56Lev5b6EXG4gICAgICAgICAgaWYgKCFleGlzdHNTeW5jKGFic3RyYWN0SW1hZ2VGaWxlKSkge1xuICAgICAgICAgICAgYWJzdHJhY3RJbWFnZUZpbGUgPSBkZWNvZGVVUkkoXG4gICAgICAgICAgICAgIGpvaW4oXG4gICAgICAgICAgICAgICAgYmFzZVBhdGgsXG4gICAgICAgICAgICAgICAgcG9zaXgucmVzb2x2ZShwb3NpeC5qb2luKFwiL1wiLCB0aGlzUGF0aC5wYXJlbnQucGF0aCksIGVuY29kZWRVcmkpXG4gICAgICAgICAgICAgIClcbiAgICAgICAgICAgICk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG5cbiAgICAgICAgaWYgKFxuICAgICAgICAgIGV4aXN0c1N5bmMoYWJzdHJhY3RJbWFnZUZpbGUpICYmXG4gICAgICAgICAgaXNBc3NldFR5cGVBbkltYWdlKGFic3RyYWN0SW1hZ2VGaWxlKVxuICAgICAgICApIHtcbiAgICAgICAgICBpbWFnZUxpc3QucHVzaCh7XG4gICAgICAgICAgICBwYXRoOiBhYnN0cmFjdEltYWdlRmlsZSxcbiAgICAgICAgICAgIG5hbWU6IGltYWdlTmFtZSxcbiAgICAgICAgICAgIHNvdXJjZTogbWF0Y2guc291cmNlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIGlmIChpbWFnZUxpc3QubGVuZ3RoID09PSAwKSB7XG4gICAgICBuZXcgTm90aWNlKFwi5rKh5pyJ6Kej5p6Q5Yiw5Zu+5YOP5paH5Lu2XCIpO1xuICAgICAgcmV0dXJuO1xuICAgIH0gZWxzZSB7XG4gICAgICBuZXcgTm90aWNlKGDlhbHmib7liLAke2ltYWdlTGlzdC5sZW5ndGh95Liq5Zu+5YOP5paH5Lu277yM5byA5aeL5LiK5LygYCk7XG4gICAgfVxuICAgIGNvbnNvbGUubG9nKGltYWdlTGlzdCk7XG5cbiAgICB0aGlzLnVwbG9hZGVyLnVwbG9hZEZpbGVzKGltYWdlTGlzdC5tYXAoaXRlbSA9PiBpdGVtLnBhdGgpKS50aGVuKHJlcyA9PiB7XG4gICAgICBpZiAocmVzLnN1Y2Nlc3MpIHtcbiAgICAgICAgbGV0IHVwbG9hZFVybExpc3QgPSByZXMucmVzdWx0O1xuICAgICAgICBpbWFnZUxpc3QubWFwKGl0ZW0gPT4ge1xuICAgICAgICAgIC8vIGdpdGVh5LiN6IO95LiK5Lyg6LaF6L+HMU3nmoTmlbDmja7vvIzkuIrkvKDlpJrlvKDnhafniYfvvIzplJnor6/nmoTor53kvJrov5Tlm57ku4DkuYjvvJ/ov5jmnInlvoXpqozor4FcbiAgICAgICAgICBjb25zdCB1cGxvYWRJbWFnZSA9IHVwbG9hZFVybExpc3Quc2hpZnQoKTtcbiAgICAgICAgICBrZXkgPSBrZXkucmVwbGFjZUFsbChpdGVtLnNvdXJjZSwgYCFbJHtpdGVtLm5hbWV9XSgke3VwbG9hZEltYWdlfSlgKTtcbiAgICAgICAgfSk7XG4gICAgICAgIHRoaXMuaGVscGVyLnNldFZhbHVlKGtleSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBuZXcgTm90aWNlKFwiVXBsb2FkIGVycm9yXCIpO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgc2V0dXBQYXN0ZUhhbmRsZXIoKSB7XG4gICAgdGhpcy5yZWdpc3RlckV2ZW50KFxuICAgICAgdGhpcy5hcHAud29ya3NwYWNlLm9uKFxuICAgICAgICBcImVkaXRvci1wYXN0ZVwiLFxuICAgICAgICAoZXZ0OiBDbGlwYm9hcmRFdmVudCwgZWRpdG9yOiBFZGl0b3IsIG1hcmtkb3duVmlldzogTWFya2Rvd25WaWV3KSA9PiB7XG4gICAgICAgICAgY29uc3QgYWxsb3dVcGxvYWQgPSB0aGlzLmhlbHBlci5nZXRGcm9udG1hdHRlclZhbHVlKFxuICAgICAgICAgICAgXCJpbWFnZS1hdXRvLXVwbG9hZFwiLFxuICAgICAgICAgICAgdGhpcy5zZXR0aW5ncy51cGxvYWRCeUNsaXBTd2l0Y2hcbiAgICAgICAgICApO1xuXG4gICAgICAgICAgbGV0IGZpbGVzID0gZXZ0LmNsaXBib2FyZERhdGEuZmlsZXM7XG4gICAgICAgICAgaWYgKCFhbGxvd1VwbG9hZCkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgIH1cbiAgICAgICAgICAvLyDliarotLTmnb/lhoXlrrnmnIltZOagvOW8j+eahOWbvueJh+aXtlxuICAgICAgICAgIGlmICh0aGlzLnNldHRpbmdzLndvcmtPbk5ldFdvcmspIHtcbiAgICAgICAgICAgIGNvbnN0IGNsaXBib2FyZFZhbHVlID0gZXZ0LmNsaXBib2FyZERhdGEuZ2V0RGF0YShcInRleHQvcGxhaW5cIik7XG4gICAgICAgICAgICBjb25zdCBpbWFnZUxpc3QgPSB0aGlzLmhlbHBlclxuICAgICAgICAgICAgICAuZ2V0SW1hZ2VMaW5rKGNsaXBib2FyZFZhbHVlKVxuICAgICAgICAgICAgICAuZmlsdGVyKGltYWdlID0+IGltYWdlLnBhdGguc3RhcnRzV2l0aChcImh0dHBcIikpO1xuXG4gICAgICAgICAgICBpZiAoaW1hZ2VMaXN0Lmxlbmd0aCAhPT0gMCkge1xuICAgICAgICAgICAgICB0aGlzLnVwbG9hZGVyXG4gICAgICAgICAgICAgICAgLnVwbG9hZEZpbGVzKGltYWdlTGlzdC5tYXAoaXRlbSA9PiBpdGVtLnBhdGgpKVxuICAgICAgICAgICAgICAgIC50aGVuKHJlcyA9PiB7XG4gICAgICAgICAgICAgICAgICBsZXQgdmFsdWUgPSB0aGlzLmhlbHBlci5nZXRWYWx1ZSgpO1xuICAgICAgICAgICAgICAgICAgaWYgKHJlcy5zdWNjZXNzKSB7XG4gICAgICAgICAgICAgICAgICAgIGxldCB1cGxvYWRVcmxMaXN0ID0gcmVzLnJlc3VsdDtcbiAgICAgICAgICAgICAgICAgICAgaW1hZ2VMaXN0Lm1hcChpdGVtID0+IHtcbiAgICAgICAgICAgICAgICAgICAgICBjb25zdCB1cGxvYWRJbWFnZSA9IHVwbG9hZFVybExpc3Quc2hpZnQoKTtcbiAgICAgICAgICAgICAgICAgICAgICB2YWx1ZSA9IHZhbHVlLnJlcGxhY2VBbGwoXG4gICAgICAgICAgICAgICAgICAgICAgICBpdGVtLnNvdXJjZSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGAhWyR7aXRlbS5uYW1lfV0oJHt1cGxvYWRJbWFnZX0pYFxuICAgICAgICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgICAgICB0aGlzLmhlbHBlci5zZXRWYWx1ZSh2YWx1ZSk7XG4gICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBuZXcgTm90aWNlKFwiVXBsb2FkIGVycm9yXCIpO1xuICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cblxuICAgICAgICAgIC8vIOWJqui0tOadv+S4reaYr+WbvueJh+aXtui/m+ihjOS4iuS8oFxuICAgICAgICAgIGlmIChcbiAgICAgICAgICAgIGlzQ29weUltYWdlRmlsZSgpIHx8XG4gICAgICAgICAgICBmaWxlcy5sZW5ndGggIT09IDAgfHxcbiAgICAgICAgICAgIGZpbGVzWzBdPy50eXBlLnN0YXJ0c1dpdGgoXCJpbWFnZVwiKVxuICAgICAgICAgICkge1xuICAgICAgICAgICAgdGhpcy51cGxvYWRGaWxlQW5kRW1iZWRJbWd1ckltYWdlKFxuICAgICAgICAgICAgICBlZGl0b3IsXG4gICAgICAgICAgICAgIGFzeW5jIChlZGl0b3I6IEVkaXRvciwgcGFzdGVJZDogc3RyaW5nKSA9PiB7XG4gICAgICAgICAgICAgICAgbGV0IHJlcyA9IGF3YWl0IHRoaXMudXBsb2FkZXIudXBsb2FkRmlsZUJ5Q2xpcGJvYXJkKCk7XG5cbiAgICAgICAgICAgICAgICBpZiAocmVzLmNvZGUgIT09IDApIHtcbiAgICAgICAgICAgICAgICAgIHRoaXMuaGFuZGxlRmFpbGVkVXBsb2FkKGVkaXRvciwgcGFzdGVJZCwgcmVzLm1zZyk7XG4gICAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGNvbnN0IHVybCA9IHJlcy5kYXRhO1xuICAgICAgICAgICAgICAgIHJldHVybiB1cmw7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICkuY2F0Y2goY29uc29sZS5lcnJvcik7XG4gICAgICAgICAgICBldnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIClcbiAgICApO1xuICAgIHRoaXMucmVnaXN0ZXJFdmVudChcbiAgICAgIHRoaXMuYXBwLndvcmtzcGFjZS5vbihcbiAgICAgICAgXCJlZGl0b3ItZHJvcFwiLFxuICAgICAgICBhc3luYyAoZXZ0OiBEcmFnRXZlbnQsIGVkaXRvcjogRWRpdG9yLCBtYXJrZG93blZpZXc6IE1hcmtkb3duVmlldykgPT4ge1xuICAgICAgICAgIGNvbnN0IGFsbG93VXBsb2FkID0gdGhpcy5oZWxwZXIuZ2V0RnJvbnRtYXR0ZXJWYWx1ZShcbiAgICAgICAgICAgIFwiaW1hZ2UtYXV0by11cGxvYWRcIixcbiAgICAgICAgICAgIHRoaXMuc2V0dGluZ3MudXBsb2FkQnlDbGlwU3dpdGNoXG4gICAgICAgICAgKTtcbiAgICAgICAgICBsZXQgZmlsZXMgPSBldnQuZGF0YVRyYW5zZmVyLmZpbGVzO1xuXG4gICAgICAgICAgaWYgKCFhbGxvd1VwbG9hZCkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGlmIChmaWxlcy5sZW5ndGggIT09IDAgJiYgZmlsZXNbMF0udHlwZS5zdGFydHNXaXRoKFwiaW1hZ2VcIikpIHtcbiAgICAgICAgICAgIGxldCBzZW5kRmlsZXM6IEFycmF5PFN0cmluZz4gPSBbXTtcbiAgICAgICAgICAgIGxldCBmaWxlcyA9IGV2dC5kYXRhVHJhbnNmZXIuZmlsZXM7XG4gICAgICAgICAgICBBcnJheS5mcm9tKGZpbGVzKS5mb3JFYWNoKChpdGVtLCBpbmRleCkgPT4ge1xuICAgICAgICAgICAgICBzZW5kRmlsZXMucHVzaChpdGVtLnBhdGgpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBldnQucHJldmVudERlZmF1bHQoKTtcblxuICAgICAgICAgICAgY29uc3QgZGF0YSA9IGF3YWl0IHRoaXMudXBsb2FkZXIudXBsb2FkRmlsZXMoc2VuZEZpbGVzKTtcblxuICAgICAgICAgICAgaWYgKGRhdGEuc3VjY2Vzcykge1xuICAgICAgICAgICAgICBkYXRhLnJlc3VsdC5tYXAoKHZhbHVlOiBzdHJpbmcpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgcGFzdGVJZCA9IChNYXRoLnJhbmRvbSgpICsgMSkudG9TdHJpbmcoMzYpLnN1YnN0cigyLCA1KTtcbiAgICAgICAgICAgICAgICB0aGlzLmluc2VydFRlbXBvcmFyeVRleHQoZWRpdG9yLCBwYXN0ZUlkKTtcbiAgICAgICAgICAgICAgICB0aGlzLmVtYmVkTWFya0Rvd25JbWFnZShlZGl0b3IsIHBhc3RlSWQsIHZhbHVlKTtcbiAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICBuZXcgTm90aWNlKFwiVXBsb2FkIGVycm9yXCIpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgKVxuICAgICk7XG4gIH1cblxuICBhc3luYyB1cGxvYWRGaWxlQW5kRW1iZWRJbWd1ckltYWdlKGVkaXRvcjogRWRpdG9yLCBjYWxsYmFjazogRnVuY3Rpb24pIHtcbiAgICBsZXQgcGFzdGVJZCA9IChNYXRoLnJhbmRvbSgpICsgMSkudG9TdHJpbmcoMzYpLnN1YnN0cigyLCA1KTtcbiAgICB0aGlzLmluc2VydFRlbXBvcmFyeVRleHQoZWRpdG9yLCBwYXN0ZUlkKTtcblxuICAgIHRyeSB7XG4gICAgICBjb25zdCB1cmwgPSBhd2FpdCBjYWxsYmFjayhlZGl0b3IsIHBhc3RlSWQpO1xuICAgICAgdGhpcy5lbWJlZE1hcmtEb3duSW1hZ2UoZWRpdG9yLCBwYXN0ZUlkLCB1cmwpO1xuICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgIHRoaXMuaGFuZGxlRmFpbGVkVXBsb2FkKGVkaXRvciwgcGFzdGVJZCwgZSk7XG4gICAgfVxuICB9XG5cbiAgaW5zZXJ0VGVtcG9yYXJ5VGV4dChlZGl0b3I6IEVkaXRvciwgcGFzdGVJZDogc3RyaW5nKSB7XG4gICAgbGV0IHByb2dyZXNzVGV4dCA9IGltYWdlQXV0b1VwbG9hZFBsdWdpbi5wcm9ncmVzc1RleHRGb3IocGFzdGVJZCk7XG4gICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24ocHJvZ3Jlc3NUZXh0ICsgXCJcXG5cIik7XG4gIH1cblxuICBwcml2YXRlIHN0YXRpYyBwcm9ncmVzc1RleHRGb3IoaWQ6IHN0cmluZykge1xuICAgIHJldHVybiBgIVtVcGxvYWRpbmcgZmlsZS4uLiR7aWR9XSgpYDtcbiAgfVxuXG4gIGVtYmVkTWFya0Rvd25JbWFnZShlZGl0b3I6IEVkaXRvciwgcGFzdGVJZDogc3RyaW5nLCBpbWFnZVVybDogYW55KSB7XG4gICAgbGV0IHByb2dyZXNzVGV4dCA9IGltYWdlQXV0b1VwbG9hZFBsdWdpbi5wcm9ncmVzc1RleHRGb3IocGFzdGVJZCk7XG4gICAgbGV0IG1hcmtEb3duSW1hZ2UgPSBgIVtdKCR7aW1hZ2VVcmx9KWA7XG5cbiAgICBpbWFnZUF1dG9VcGxvYWRQbHVnaW4ucmVwbGFjZUZpcnN0T2NjdXJyZW5jZShcbiAgICAgIGVkaXRvcixcbiAgICAgIHByb2dyZXNzVGV4dCxcbiAgICAgIG1hcmtEb3duSW1hZ2VcbiAgICApO1xuICB9XG5cbiAgaGFuZGxlRmFpbGVkVXBsb2FkKGVkaXRvcjogRWRpdG9yLCBwYXN0ZUlkOiBzdHJpbmcsIHJlYXNvbjogYW55KSB7XG4gICAgY29uc29sZS5lcnJvcihcIkZhaWxlZCByZXF1ZXN0OiBcIiwgcmVhc29uKTtcbiAgICBsZXQgcHJvZ3Jlc3NUZXh0ID0gaW1hZ2VBdXRvVXBsb2FkUGx1Z2luLnByb2dyZXNzVGV4dEZvcihwYXN0ZUlkKTtcbiAgICBpbWFnZUF1dG9VcGxvYWRQbHVnaW4ucmVwbGFjZUZpcnN0T2NjdXJyZW5jZShcbiAgICAgIGVkaXRvcixcbiAgICAgIHByb2dyZXNzVGV4dCxcbiAgICAgIFwi4pqg77iPdXBsb2FkIGZhaWxlZCwgY2hlY2sgZGV2IGNvbnNvbGVcIlxuICAgICk7XG4gIH1cblxuICBzdGF0aWMgcmVwbGFjZUZpcnN0T2NjdXJyZW5jZShcbiAgICBlZGl0b3I6IEVkaXRvcixcbiAgICB0YXJnZXQ6IHN0cmluZyxcbiAgICByZXBsYWNlbWVudDogc3RyaW5nXG4gICkge1xuICAgIGxldCBsaW5lcyA9IGVkaXRvci5nZXRWYWx1ZSgpLnNwbGl0KFwiXFxuXCIpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbGluZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGxldCBjaCA9IGxpbmVzW2ldLmluZGV4T2YodGFyZ2V0KTtcbiAgICAgIGlmIChjaCAhPSAtMSkge1xuICAgICAgICBsZXQgZnJvbSA9IHsgbGluZTogaSwgY2g6IGNoIH07XG4gICAgICAgIGxldCB0byA9IHsgbGluZTogaSwgY2g6IGNoICsgdGFyZ2V0Lmxlbmd0aCB9O1xuICAgICAgICBlZGl0b3IucmVwbGFjZVJhbmdlKHJlcGxhY2VtZW50LCBmcm9tLCB0byk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgIH1cbiAgfVxufVxuIl0sIm5hbWVzIjpbInBhdGgiLCJleHRuYW1lIiwiY2xpcGJvYXJkIiwicmVxdWVzdFVybCIsIk5vdGljZSIsImV4ZWMiLCJNYXJrZG93blZpZXciLCJwYXJzZSIsIlNldHRpbmciLCJQbHVnaW5TZXR0aW5nVGFiIiwiYWRkSWNvbiIsImV4aXN0c1N5bmMiLCJta2RpclN5bmMiLCJqb2luIiwicmVzb2x2ZSIsIm5vcm1hbGl6ZVBhdGgiLCJyZWxhdGl2ZSIsIndyaXRlRmlsZVN5bmMiLCJURmlsZSIsInBvc2l4IiwiUGx1Z2luIl0sIm1hcHBpbmdzIjoiOzs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsSUFBSSxhQUFhLEdBQUcsU0FBUyxDQUFDLEVBQUUsQ0FBQyxFQUFFO0FBQ25DLElBQUksYUFBYSxHQUFHLE1BQU0sQ0FBQyxjQUFjO0FBQ3pDLFNBQVMsRUFBRSxTQUFTLEVBQUUsRUFBRSxFQUFFLFlBQVksS0FBSyxJQUFJLFVBQVUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxTQUFTLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNwRixRQUFRLFVBQVUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEtBQUssSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLElBQUksTUFBTSxDQUFDLFNBQVMsQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUMxRyxJQUFJLE9BQU8sYUFBYSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztBQUMvQixDQUFDLENBQUM7QUFDRjtBQUNPLFNBQVMsU0FBUyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUU7QUFDaEMsSUFBSSxJQUFJLE9BQU8sQ0FBQyxLQUFLLFVBQVUsSUFBSSxDQUFDLEtBQUssSUFBSTtBQUM3QyxRQUFRLE1BQU0sSUFBSSxTQUFTLENBQUMsc0JBQXNCLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLCtCQUErQixDQUFDLENBQUM7QUFDbEcsSUFBSSxhQUFhLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0FBQ3hCLElBQUksU0FBUyxFQUFFLEdBQUcsRUFBRSxJQUFJLENBQUMsV0FBVyxHQUFHLENBQUMsQ0FBQyxFQUFFO0FBQzNDLElBQUksQ0FBQyxDQUFDLFNBQVMsR0FBRyxDQUFDLEtBQUssSUFBSSxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLFNBQVMsR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQztBQUN6RixDQUFDO0FBdUNEO0FBQ08sU0FBUyxTQUFTLENBQUMsT0FBTyxFQUFFLFVBQVUsRUFBRSxDQUFDLEVBQUUsU0FBUyxFQUFFO0FBQzdELElBQUksU0FBUyxLQUFLLENBQUMsS0FBSyxFQUFFLEVBQUUsT0FBTyxLQUFLLFlBQVksQ0FBQyxHQUFHLEtBQUssR0FBRyxJQUFJLENBQUMsQ0FBQyxVQUFVLE9BQU8sRUFBRSxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFO0FBQ2hILElBQUksT0FBTyxLQUFLLENBQUMsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDLEVBQUUsVUFBVSxPQUFPLEVBQUUsTUFBTSxFQUFFO0FBQy9ELFFBQVEsU0FBUyxTQUFTLENBQUMsS0FBSyxFQUFFLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRTtBQUNuRyxRQUFRLFNBQVMsUUFBUSxDQUFDLEtBQUssRUFBRSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsU0FBUyxDQUFDLE9BQU8sQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRTtBQUN0RyxRQUFRLFNBQVMsSUFBSSxDQUFDLE1BQU0sRUFBRSxFQUFFLE1BQU0sQ0FBQyxJQUFJLEdBQUcsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsR0FBRyxLQUFLLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLENBQUMsRUFBRTtBQUN0SCxRQUFRLElBQUksQ0FBQyxDQUFDLFNBQVMsR0FBRyxTQUFTLENBQUMsS0FBSyxDQUFDLE9BQU8sRUFBRSxVQUFVLElBQUksRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQztBQUM5RSxLQUFLLENBQUMsQ0FBQztBQUNQLENBQUM7QUFDRDtBQUNPLFNBQVMsV0FBVyxDQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUU7QUFDM0MsSUFBSSxJQUFJLENBQUMsR0FBRyxFQUFFLEtBQUssRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLFdBQVcsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsSUFBSSxFQUFFLEVBQUUsRUFBRSxHQUFHLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDO0FBQ3JILElBQUksT0FBTyxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLE9BQU8sTUFBTSxLQUFLLFVBQVUsS0FBSyxDQUFDLENBQUMsTUFBTSxDQUFDLFFBQVEsQ0FBQyxHQUFHLFdBQVcsRUFBRSxPQUFPLElBQUksQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUM7QUFDN0osSUFBSSxTQUFTLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxPQUFPLFVBQVUsQ0FBQyxFQUFFLEVBQUUsT0FBTyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRTtBQUN0RSxJQUFJLFNBQVMsSUFBSSxDQUFDLEVBQUUsRUFBRTtBQUN0QixRQUFRLElBQUksQ0FBQyxFQUFFLE1BQU0sSUFBSSxTQUFTLENBQUMsaUNBQWlDLENBQUMsQ0FBQztBQUN0RSxRQUFRLE9BQU8sQ0FBQyxFQUFFLElBQUk7QUFDdEIsWUFBWSxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxRQUFRLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksRUFBRSxPQUFPLENBQUMsQ0FBQztBQUN6SyxZQUFZLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDcEQsWUFBWSxRQUFRLEVBQUUsQ0FBQyxDQUFDLENBQUM7QUFDekIsZ0JBQWdCLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLE1BQU07QUFDOUMsZ0JBQWdCLEtBQUssQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksRUFBRSxLQUFLLEVBQUUsQ0FBQztBQUN4RSxnQkFBZ0IsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUztBQUNqRSxnQkFBZ0IsS0FBSyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsU0FBUztBQUNqRSxnQkFBZ0I7QUFDaEIsb0JBQW9CLElBQUksRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLE1BQU0sR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxTQUFTLEVBQUU7QUFDaEksb0JBQW9CLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLEVBQUU7QUFDMUcsb0JBQW9CLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxD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
