function varargout = packageGUI(varargin)
% PACKAGEGUI M-file for packageGUI.fig
%      PACKAGEGUI, by itself, creates a new PACKAGEGUI or raises the existing
%      singleton*.
%
%      H = PACKAGEGUI returns the handle to a new PACKAGEGUI or the handle to
%      the existing singleton*.
%
%      PACKAGEGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PACKAGEGUI.M with the given input arguments.
%
%      PACKAGEGUI('Property','Value',...) creates a new PACKAGEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before packageGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to packageGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

% Edit the above text to modify the response to help packageGUI

% Last Modified by GUIDE v2.5 21-Jun-2019 15:56:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @packageGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @packageGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Outputs from this function are returned to the command line.
function varargout = packageGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% In case the package GUI has been called without argument
userData = get(handles.figure1, 'UserData');
if (isfield(userData,'startMovieSelectorGUI') && userData.startMovieSelectorGUI)
    movieSelectorGUI('packageName',userData.packageName,'MD',userData.MD,...
        'ML', userData.ML , 'ImD', userData.ImD, 'cluster', uTrackParCluster);
    delete(handles.figure1)
end

% --- Executes on button press in pushbutton_status.
function pushbutton_status_Callback(~, ~, handles)
userData = get(handles.figure1, 'UserData');

% if movieDataGUI exist
if isfield(userData, 'overviewFig') && ishandle(userData.overviewFig)
    delete(userData.overviewFig)
end

if ~isempty(userData.MD) && isempty(userData.ImD)
    userData.overviewFig = movieDataGUI(userData.MD(userData.id));
elseif isempty(userData.MD) && ~isempty(userData.ImD)
    userData.overviewFig = imageDataGUI(userData.ImD(userData.id));
end
set(handles.figure1, 'UserData', userData);

% --- Executes on Save button press or File>Save
function save_Callback(~, ~, handles)
userData = get(handles.figure1, 'UserData');
if ~isempty(userData.MD) && isempty(userData.ImD)
set(handles.text_saveStatus, 'Visible', 'on')
arrayfun(@save,userData.MD);
pause(.3)
set(handles.text_saveStatus, 'Visible', 'off')
elseif isempty(userData.MD) && ~isempty(userData.ImD)
    set(handles.text_ImDsaveStatus, 'Visible', 'on')
    arrayfun(@save,userData.ImD);
    pause(.3)
    set(handles.text_ImDsaveStatus, 'Visible', 'off')
end


function switchMovie_Callback(hObject, ~, handles)

userData = get(handles.figure1, 'UserData');
if ~isempty(userData.MD) && isempty(userData.ImD)
nMovies = length(userData.MD);
elseif isempty(userData.MD) && ~isempty(userData.ImD)
    nMovies = length(userData.ImD);
end

switch get(hObject,'Tag')
    case 'pushbutton_left'
        newMovieId = userData.id - 1;
    case 'pushbutton_right'
        newMovieId = userData.id + 1;
    case 'popupmenu_movie'
        newMovieId = get(hObject, 'Value');
    otherwise
end

if (newMovieId==userData.id), return; end

% Save previous movie checkboxes
userData.statusM(userData.id).Checked = userfcn_saveCheckbox(handles);

% Set up new movie GUI parameters
userData.id = mod(newMovieId-1,nMovies)+1;
if isa(userData.crtPackage, 'XcorrFluctuationPackage')
    nMovieLists = length(userData.ML);
    userData.id = mod(newMovieId-1,nMovieLists)+1;
end
userData.crtPackage = userData.package(userData.id);
set(handles.figure1, 'UserData', userData)
set(handles.popupmenu_movie, 'Value', userData.id)

% Set up GUI
if userData.statusM(userData.id).Visited
   packageGUI_RefreshFcn(handles, 'refresh') 
else
   packageGUI_RefreshFcn(handles, 'initialize') 
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
userData = get(handles.figure1,'Userdata');
if isfield(userData, 'MD') && isempty(userData.ImD)
    MD = userData.MD;
elseif isfield(userData, 'ImD') && isempty(userData.MD) && ~isempty(userData.ImD)
    ImD = userData.ImD;
else
    delete(handles.figure1);
    return;
end

saveRes = questdlg('Do you want to save the current progress?', ...
    'Package Control Panel');

if strcmpi(saveRes,'yes')
    if isfield(userData, 'MD') && isempty(userData.ImD)
        arrayfun(@save,userData.MD);
    elseif isfield(userData, 'ImD') && isempty(userData.MD) && ~isempty(userData.ImD)
        arrayfun(@save,userData.ImD);
    end
end
if strcmpi(saveRes,'cancel'), return; end
delete(handles.figure1);

% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)

userData = get(handles.figure1, 'UserData');
if ~isempty(userData)
    if ~isempty(userData.MD)
        if userData.MD.isMock()
            load([userData.MD.mockMD_.parent.movieDataPath_, filesep, userData.MD.mockMD_.parent.movieDataFileName_]);
            movieViewer(MD, 'refresher', '1');
        end
    end
end

% Find all figures stored in userData and delete them
if isempty(userData), return; end
userDataFields=fieldnames(userData);
isFig = ~cellfun(@isempty,regexp(userDataFields,'Fig$'));
userDataFigs = userDataFields(isFig);
for i=1:numel(userDataFigs)
     figHandles = userData.(userDataFigs{i});
     if ~iscell(figHandles)
     validFigHandles = figHandles(ishandle(figHandles)&figHandles ~= 0);
     delete(validFigHandles);
     else 
        % this is for new APP DESIGNER UI:
         validFigHandles = cellfun(@(x) x((ishandle(x) | isa(x, 'matlab.apps.AppBase')) & x ~= 0), figHandles, 'UniformOutput', false);
         validFigHandles = validFigHandles(~cellfun(@isempty, validFigHandles));
         cellfun(@delete, validFigHandles);
     end
end

% msgboxGUI used for error reports
if isfield(userData, 'msgboxGUI') && ishandle(userData.msgboxGUI)
   delete(userData.msgboxGUI) 
end

% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)

if strcmp(eventdata.Key, 'return')
    exit_Callback(handles.pushbutton_exit, [], handles);
end
if strcmp(eventdata.Key, 'leftarrow')
    switchMovie_Callback(handles.pushbutton_left, [], handles);
end
if strcmp(eventdata.Key, 'rightarrow')
    switchMovie_Callback(handles.pushbutton_right, [], handles);
end

% --------------------------------------------------------------------
function menu_about_Callback(hObject, eventdata, handles)

status = web(get(hObject,'UserData'), '-browser');
if status
    switch status
        case 1
            msg = 'System default web browser is not found.';
        case 2
            msg = 'System default web browser is found but could not be launched.';
        otherwise
            msg = 'Fail to open browser for unknown reason.';
    end
    warndlg(msg,'Fail to open browser','modal');
end

% --------------------------------------------------------------------
function menu_file_open_Callback(~, ~, handles)
% Call back function of 'Open' in menu bar
userData = get(handles.figure1,'Userdata');
% if ~isempty(userData.MD), field = 'MD'; else field = 'ML'; end
% arrayfun(@(x) x.save,userData.(field));
movieSelectorGUI('packageName',userData.packageName,...
    'MD', userData.MD, 'ML', userData.ML, 'ImD', userData.ImD);
delete(handles.figure1)

% --------------------------------------------------------------------
function exit_Callback(~, ~, handles)

delete(handles.figure1);


% --- Executes on button press in pushbutton_show.
function pushbutton_show_Callback(hObject, ~, handles)

userData = get(handles.figure1, 'UserData');
prop=get(hObject,'Tag');
procID = str2double(prop(length('pushbutton_show_')+1:end));

if isfield(userData, 'resultFig') && ishandle(userData.resultFig)
    delete(userData.resultFig)
end

userData.resultFig = userData.crtPackage.processes_{procID}.resultDisplay();
    
set(handles.figure1, 'UserData', userData);

% --- Executes on button press in pushbutton_set.
function pushbutton_set_Callback(hObject, ~, handles)
global deactivateCLIBackup
userData = get(handles.figure1, 'UserData');
prop=get(hObject,'Tag');
procID = str2double(prop(length('pushbutton_set_')+1:end));

% Read GUI handle from the associated process static method
crtProc=userData.crtPackage.getProcessClassNames{procID};
crtProcGUI =eval([crtProc '.GUI']);
try 
    userData.setFig{procID} = crtProcGUI('mainFig',handles.figure1,procID); % Save setFig in cell array {}, since new APP DESIGNER UI has diff types
catch ME
    if exist('deactivateCLIBackup','var') == 1 && isempty(deactivateCLIBackup)
        rethrow(ME)
    else
        msgbox(ME.message)
        warning('Loading Custom GUI failed! -- Running CLI parameter config as BACKUP - follow instructions');
        uiwait(msgbox({'Loading Custom GUI failed!','Running CLI parameter config as BACKUP','Please follow instructions'}));
        userData.setFig{procID} = cliGUI('mainFig',handles.figure1,procID);
    end
end

set(handles.figure1, 'UserData', userData);
guidata(hObject,handles);

% --- Executes on button press in checkbox.
function checkbox_Callback(hObject, eventdata, handles)

props=get(hObject,{'Value','Tag'});
procStatus=props{1};
procID = str2double(props{2}(length('checkbox_')+1:end));

userData=get(handles.figure1, 'UserData');
userData.statusM(userData.id).Checked(procID) = procStatus;
set(handles.figure1, 'UserData', userData)


userfcn_checkAllMovies(procID, procStatus, handles);
userfcn_lampSwitch(procID, procStatus, handles);

% --------------------------------------------------------------------
function menu_debug_enter_Callback(hObject, eventdata, handles)

status = get(hObject,'Checked');
if strcmp(status,'on'), 
    newstatus = 'off'; 
    dbclear if caught error;
else
    newstatus='on'; 
end
set(hObject,'Checked',newstatus);


% --------------------------------------------------------------------
function menu_debug_batchMode_Callback(hObject, eventdata, handles)

status = get(hObject,'Checked');
if strcmp(status,'on'), 
    newstatus = 'off'; 
else
    newstatus='on'; 
end
set(hObject,'Checked',newstatus);


% --- Executes on button press in pushbutton_open.
function pushbutton_open_Callback(hObject, eventdata, handles)
userData = get(handles.figure1, 'UserData');
prop=get(hObject,'Tag');
procID = str2double(prop(length('pushbutton_show_')+1:end));

if ~isequal(userData.packageName, 'XcorrFluctuationPackage')
    % Use the OS-specific command to open result in exploration window
    outputDir = userData.crtPackage.processes_{procID}.funParams_.OutputDirectory;
    if ispc
        winopen(outputDir);
    elseif ismac
        system(sprintf('open %s',regexptranslate('escape',outputDir)));
    elseif isunix
        status = system(sprintf('gio open "%s"',regexptranslate('escape',outputDir)));
        % If a non-zero integer is returned, then display a message box
        if(status)
            msgbox(sprintf('Results can be found under %s',regexptranslate('escape',outputDir)));
        end
    else
        msgbox(sprintf('Results can be found under %s',regexptranslate('escape',outputDir)));
        % SB: Following command not working under Ubuntu (as well as gnome-open
        % & nautilus)
        % system(sprintf('xdg-open %s',regexptranslate('escape',outputDir)));
    end
else
    if isfield(userData, 'folderFig') && ishandle(userData.folderFig)
        delete(userData.folderFig)
    end
    userData.folderFig = userData.crtPackage.processes_{procID}.folderDisplay();
    set(handles.figure1, 'UserData', userData);
end
    


% --------------------------------------------------------------------
function menu_parallel_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% clusterMenu = findobj(hObject,'Label','Cluster');
% poolSizeMenu = findobj(hObject,'Label','Pool Size');
% batchMenu = findobj(hObject,'Label','Batch Mode');
clusterMenu = handles.menu_parallel_cluster;
poolSizeMenu = handles.menu_parallel_pool;
batchMenu = handles.menu_parallel_batch;

% Setup Cluster Menu
delete(clusterMenu.Children);
profiles = {};
try
    profiles = parallel.clusterProfiles;
catch err
    warning('Could not obtain cluster profiles');
    disp(getReport(err));
end
profiles = [{'None'} profiles];
% Update cluster from uTrackParCluster
cluster = uTrackParCluster();

for i=1:length(profiles)
    h = uimenu(clusterMenu,'Label',profiles{i},'Callback',{@menu_parallel_cluster_profile_Callback,handles});

    if(i == 1 && isempty(cluster))
        h.Checked = 'on';
    elseif(~isempty(cluster) && strcmp(profiles{i},cluster.Profile))
        h.Checked = 'on';
        if(isempty(poolSizeMenu.Children))
            menu_parallel_cluster_profile_Callback(h, eventdata, handles)
        end
    end
    if(i == 2)
        h.Separator = 'on';
    end
end

% Setup Pool Size Menu
if(isempty(cluster))
    poolSizeMenu.Enable = 'off';
    batchMenu.Enable = 'off';
else
    poolSizeMenu.Enable = 'on';
    batchMenu.Enable = 'on';
end




% --------------------------------------------------------------------
function menu_parallel_cluster_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_cluster (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



% --------------------------------------------------------------------
function menu_parallel_cluster_profile_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_cluster_profile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(strcmp(hObject.Label,'None'))
    cluster = [];
else
    cluster = parcluster(hObject.Label);
    poolSizeMenu = handles.menu_parallel_pool;
    delete(poolSizeMenu.Children);
    currentPool = gcp('nocreate');
    h = uimenu(poolSizeMenu,'Label','No Pool','Callback',{@menu_parallel_pool_size_Callback,handles});
    % Provide at most 16 options
    N = max(floor(cluster.NumWorkers/16),1);
    if(isempty(currentPool))
        if(strcmp(cluster.Profile,'local'))
            % For local profile, set the default to maximum number of
            % workers
            poolSize = cluster.NumWorkers;
        else
            % For other profiles, use the first increment
            poolSize = N;
        end
    else
        % If pool exists, display current size
        poolSize = currentPool.NumWorkers;
    end

    % List pool sizes for every N workers
    % make sure existing pool size is checked
    poolSizeCheckExists = false;
    for i=N:N:cluster.NumWorkers
        h = uimenu(poolSizeMenu,'Label',num2str(i),'Callback',{@menu_parallel_pool_size_Callback,handles});
        if(i == N)
            h.Separator = 'on';
        end
        if(i == poolSize)
            h.Checked = 'on';
            poolSizeCheckExists = true;
        end
    end
    if(~poolSizeCheckExists)
        h = uimenu(poolSizeMenu,'Label',num2str(poolSize),'Callback',{@menu_parallel_pool_size_Callback,handles});
        h.Checked = 'on';
    end
end
% Set uTrackParCluster so that this information is accessible
uTrackParCluster(cluster);
batchMenuItem = findobj(handles.menu_parallel_batch,'Checked','on');
batchMenuItem.Callback(batchMenuItem, eventdata);




% --------------------------------------------------------------------
function menu_parallel_pool_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_pool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --------------------------------------------------------------------
function menu_parallel_pool_size_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_pool_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(hObject.Parent.Children,'Checked','off')
hObject.Checked = 'on';
% if(strcmp(handles.menu_parallel_batch_client.Checked,'on'))
%     currentPool = gcp('nocreate');
%     switch(hObject.Label)
%         case 'No Pool'
%             delete(currentPool);
%         otherwise
%             poolSize = str2double(hObject.Label);
%             if(~isempty(currentPool) && currentPool.NumWorkers ~= poolSize)
%                 delete(currentPool);
%             end
%             parpool(uTrackParCluster,poolSize);
%     end
% else
% end

% --------------------------------------------------------------------
function menu_parallel_batch_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_batch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_parallel_batch_single_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_batch_single (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hObject.Checked = 'on';
handles.menu_parallel_batch_client.Checked = 'off';
handles.menu_parallel_batch_one_per_movie.Checked = 'off';
noPool = findobj(handles.menu_parallel_pool,'Label','No Pool');
noPool.Enable = 'on';

% --------------------------------------------------------------------
function menu_parallel_batch_one_per_movie_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_batch_one_per_movie (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hObject.Checked = 'on';
handles.menu_parallel_batch_client.Checked = 'off';
handles.menu_parallel_batch_single.Checked = 'off';
noPool = findobj(handles.menu_parallel_pool,'Label','No Pool');
noPool.Enable = 'on';


% --------------------------------------------------------------------
function menu_parallel_batch_client_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_batch_client (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hObject.Checked = 'on';
handles.menu_parallel_batch_single.Checked = 'off';
handles.menu_parallel_batch_one_per_movie.Checked = 'off';
noPool = findobj(handles.menu_parallel_pool,'Label','No Pool');
noPool.Enable = 'off';
poolSize = findobj(handles.menu_parallel_pool,'Checked','on');
% No pool is not an option for This Client non-batch mode
if(strcmp(poolSize.Label,'No Pool'))
    clusterProfileMenuItem = findobj(handles.menu_parallel_cluster,'Checked','on');
    menu_parallel_cluster_profile_Callback(clusterProfileMenuItem,eventdata,handles);
end


% --------------------------------------------------------------------
function menu_parallel_help_Callback(hObject, eventdata, handles)
% hObject    handle to menu_parallel_help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h = msgbox( {...
    'uTrack Parallel Help' ...
    '--------------------' ...
    'This menu enables use of MATLAB''s parallel computing toolbox.' ...
    '' ...
    'Cluster: Select the cluster profile to use for parallezation.' ...
    '         None: Do not use parallel processing.' ...
    '         Local: Use multiprocessor capabilities of your current machine.' ...
    '         Other: Ask your cluster/HPC administrator.' ...
    '' ...
    'Pool: Select the number of workers to use. Function depends on batch mode.' ...
    '' ...
    'Batch: Select mode of parallelization.' ...
    '       This Client: Use parallel pool from this client and to evaluate' ...
    '                    movies in parallel.' ...
    '                    You may need to be on a cluster  node to ' ... 
    '                    use this if not using the local cluster.' ...
    '       Single Job: Use a single batch job to run movies(s). ' ...
    '                   Movies will be run in parallel if a pool is created.' ...
    '                   If you exit uTrack, the job will keep running.' ...
    '       One Job Per Movie: Create a batch job for each movie. ' ...
    '                          Each job will have it''s own pool.' ...
    '                          This allows processes to use the parallel pool.' ...
    '                          If you exit uTrack, the jobs will keep running.' ...
    }, ...
    'uTrack Parallel Help','help');


% --- Executes on button press in checkbox_tagName.
function checkbox_tagName_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_tagName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_tagName
% handles.processTagLabels = findall(0,'-regexp','Tag', 'processTagLabel');
% handles.processTagLabels;

%% TODO - need refresh of tags after runnings

if handles.checkbox_tagName.Value == 0;
    set(handles.processTagLabels, 'Visible', 'off')
else
    set(handles.processTagLabels, 'Visible', 'on')
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
slider_value = get(hObject, 'Value');
panel_proc_pos = get(handles.panel_proc, 'Position');
figure1_pos = get(handles.figure1, 'Position');
panel_movie_pos = get(handles.panel_movie, 'Position');

 set(handles.panel_proc, 'Position', [panel_proc_pos(1), ...
    (figure1_pos(4)-panel_movie_pos(4)-66.8-panel_proc_pos(4))*slider_value+66.8, ...
    panel_proc_pos(3), panel_proc_pos(4)]);


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
