function packageGUI_RunFcn(hObject,eventdata,handles)
% Run the selected processes in the packageGUI interface
%
% This is a common section of code called by pushbutton_run_Callback
% when user click the "Run" button on package control panels.
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

% Chuangang Ren 11/2010
% Sebastien Besson 5/2011 (last modified Oct 2011)

ip = inputParser;
ip.addRequired('hObject',@ishandle);
ip.addRequired('eventdata',@(x) isstruct(x) || isempty(x) || isa(x, 'event.EventData'));
ip.addRequired('handles',@isstruct);
ip.parse(hObject,eventdata,handles);

%% Initialization

% Get check box status of current movie and update user data
userData = get(handles.figure1,'UserData');
userData.statusM(userData.id).Checked = userfcn_saveCheckbox(handles);
set(handles.figure1, 'UserData', userData)

% Determine the movie(s) to be processed
if ~isempty(userData.MD) && isempty(userData.ImD) 
    field='MD'; 
elseif isempty(userData.MD) && ~isempty(userData.ImD) 
    field = 'ImD'; 
else
    field = 'ML'; 
end
if isa(userData.crtPackage, 'XcorrFluctuationPackage')
    field = 'ML';
end
nMovies = length(userData.(field)); % number of movies
if get(handles.checkbox_runall, 'Value')
    movieList = circshift(1:nMovies,[0 -(userData.id-1)]);
else
    movieList=userData.id;
end
% Get the list of valid movies (with processes to run)
hasValidProc = arrayfun(@(x) any(userData.statusM(x).Checked),movieList);
movieRun=movieList(hasValidProc);

procCheck=cell(1,numel(nMovies));
procCheck(movieRun)=arrayfun(@(x) find(userData.statusM(x).Checked),movieRun,...
    'UniformOutput',false);

% Throw warning dialog if no movie
if isempty(movieRun)
    warndlg('No step is selected, please select a step to process.',...
        'No Step Selected','modal');
    return
end

%% Pre-processing examination

% movie exception (same length of movie data)
movieException = cell(1, nMovies);
procRun = cell(1, nMovies);%  id of processes to run

% Find unset processes
isProcSet=@(x,y)~isempty(userData.package(x).processes_{y});
isMovieProcSet = @(x) all(arrayfun(@(y)isProcSet(x,y),procCheck{x}));
invalidMovies=movieRun(~arrayfun(isMovieProcSet,movieRun));

for i = invalidMovies
    invalidProc = procCheck{i}(arrayfun(@(y)~isProcSet(i,y),procCheck{i}));
    for j=invalidProc
        ME = MException('lccb:run:setup', ['Step %d : %s is not set up yet.\n'...
            '\nTip: when step is set up successfully, the step name becomes bold.'],j,...
            eval([userData.package(i).getProcessClassNames{j} '.getName']));
        movieException{i} = horzcat(movieException{i}, ME);
    end
end

validMovies=movieRun(arrayfun(isMovieProcSet,movieRun));
for iMovie = validMovies   
    % Check if selected processes have alrady be successfully run
    % If force run, re-run every process that is checked
    if ~get(handles.checkbox_forcerun, 'Value')
        
        k = true;
        for i = procCheck{iMovie}
            
            if  ~( userData.package(iMovie).processes_{i}.success_ && ...
                    ~userData.package(iMovie).processes_{i}.procChanged_ ) || ...
                    ~userData.package(iMovie).processes_{i}.updated_
                
                k = false;
                procRun{iMovie} = horzcat(procRun{iMovie}, i);
            end
        end
        if k
            movieRun = setdiff(movieRun, iMovie);
            continue
        end
    else
        procRun{iMovie} = procCheck{iMovie};
    end    
    
    % Package full sanity check. Sanitycheck every checked process
    [status procEx] = userData.package(iMovie).sanityCheck(true, procRun{iMovie});
    
    % Return user data !!!
    set(handles.figure1, 'UserData', userData)
    invalidProcEx = procRun{iMovie}(~cellfun(@isempty,procEx(procRun{iMovie})));
    for i = invalidProcEx
        % Check if there is fatal error in exception array
        if strcmp(procEx{i}(1).identifier, 'lccb:set:fatal') || ...
                strcmp(procEx{i}(1).identifier, 'lccb:input:fatal')
            
            % Sanity check error - switch GUI to the x th movie
            if iMovie ~= userData.id
                set(handles.popupmenu_movie, 'Value', iMovie)
                % Update the movie pop-up menu in the main package GUI
                packageGUI('switchMovie_Callback',handles.popupmenu_movie, [], handles)
            end
            
            userfcn_drawIcon(handles,'error', i, procEx{i}(1).message, true);
            
            ME = MException('lccb:run:sanitycheck','Step %d %s: \n%s',...
                i,userData.package(iMovie).processes_{i}.getName, procEx{i}(1).message);
            movieException{iMovie} = horzcat(movieException{iMovie}, ME);
                
        end
    end
    
    % Refresh user data !!!
    userData = get(handles.figure1, 'UserData');
end

%% Pre-processing exception report
if isempty(movieRun)
    warndlg('All selected steps have been processed successfully. Please check the ''Force Run'' check box if you want to re-process the successful steps.','No Step Selected','modal');
    return
end

status = generateReport(movieException,userData,'preprocessing');
if ~status, return; end

%% Start processing
if strcmp(get(handles.menu_debug_enter,'Checked'),'on'), dbstop if caught error; end

if(~isempty(uTrackParCluster))
    movieException = start_processing_movies_in_parallel(movieRun,handles,movieException,procRun);
else
    movieException = start_processing_movies_in_series(movieRun,handles,movieException,procRun);
end
if strcmp(get(handles.menu_debug_enter,'Checked'),'on'), dbclear if caught error; end
% Update userData after processing
userData = get(handles.figure1,'UserData');

%% Post-processing exception report
status = generateReport(movieException,userData,'postprocessing');
if status
    successMsg = 'Your movie(s) have been processed successfully.';
    userData.iconHelpFig = helpdlg(successMsg, [userData.crtPackage.getName]);
    set(handles.figure1, 'UserData', userData)
end

% Delete waitbars
hWaitbar = findall(0,'type','figure','tag','TMWWaitbar');
delete(hWaitbar);
end

function [movieException] = start_processing_movies_in_series(movieRun,handles,movieException, procRun)
% start_processing_movies_in_series Run legacy code for single thread
% processing

userData = get(handles.figure1, 'UserData');

for i=1:length(movieRun)
    iMovie = movieRun(i);
   
    % TODO: Consider using selectMovie here
    if iMovie ~= userData.id
        % Update the movie pop-up menu in the main package GUI
        set(handles.figure1, 'UserData', userData)
        set(handles.popupmenu_movie, 'Value', iMovie)
        
        % Update the movie pop-up menu in the main package GUI
        packageGUI('switchMovie_Callback',handles.popupmenu_movie, [], handles)
        userData = get(handles.figure1, 'UserData');
    end
    
    % Clear icons of selected processes
    % Return user data !!!
    set(handles.figure1, 'UserData', userData)
    userfcn_drawIcon(handles,'clear',procRun{iMovie},'',true); % user data is retrieved, updated and submitted
    % Refresh user data !!!
    userData = get(handles.figure1, 'UserData');
    set(handles.text_status, 'Visible', 'on');
    
    % Run algorithms!
    try
        % Return user data !!!
        set(handles.figure1, 'UserData', userData)
        
        for procID = procRun{iMovie}
            set(handles.text_status, 'String', ...
                sprintf('Step %d - Processing %d of %d movies total ...', procID, i, length(movieRun)) )
            userfcn_runProc_dfs(procID, procRun{iMovie}, handles); % user data is retrieved, updated and submitted
        end
        
    catch ME
        
        % Save the error into movie Exception cell array
        ME2 = MException('lccb:run:error','Step %d: %s',...
            procID,userData.package(iMovie).processes_{procID}.getName);
        movieException{iMovie} = horzcat(movieException{iMovie}, ME2);
        movieException{iMovie}=movieException{iMovie}.addCause(ME);
        
        procRun{iMovie} = procRun{iMovie}(procRun{iMovie} < procID);
        
        % Refresh wall status
        packageGUI_RefreshFcn(handles,'initialize');
    end
    
    % Refresh user data !!!
    userData = get(handles.figure1, 'UserData');
    set(handles.pushbutton_run, 'Enable', 'on')
    set(handles.checkbox_forcerun, 'Enable', 'on')
    set(handles.checkbox_runall, 'Enable', 'on')
    set(handles.text_status, 'Visible', 'off')

    % Return user data !!!
    set(handles.figure1, 'UserData', userData)
end
if strcmp(get(handles.menu_debug_enter,'Checked'),'on'), dbclear if caught error; end
end

function movieException = start_processing_movies_in_parallel(movieRun,handles,movieException, procRun)
% start_processing_movies_in_parallel New code for parallel processing on
% the MovieData level
    
    parSettings.batch = strcmp(handles.menu_parallel_batch_client.Checked,'on') + ...
                        strcmp(handles.menu_parallel_batch_single.Checked,'on')*2 + ...
                        strcmp(handles.menu_parallel_batch_one_per_movie.Checked,'on')*4;
    noPool = findobj(handles.menu_parallel_pool.Children,'Label','No Pool');
    parSettings.pool = ~strcmp(noPool.Checked,'on');
    
    switch(parSettings.batch)
        case 1 % This Client
            if(parSettings.pool)
                start_processing_movies_in_parallel_parfeval(movieRun,handles,movieException, procRun);
            else
                error('packageGUI_RunFcn:parallel:NotPossible', ...
                    'For running from this client, you must select a pool size');
            end
        case 2 % Single Batch Job
            start_processing_movies_in_parallel_single_batch(movieRun,handles,movieException, procRun);
        case 4 % One Per Movie
            start_processing_movies_in_parallel_batch_per_movie(movieRun,handles,movieException, procRun)
        otherwise
            error('packageGUI_RunFcn:parallel:UnknownBatchSetting', ...
                'Could not determine parallel batch setting for uTrack')
    end
    
    if(strcmp(handles.menu_parallel_batch_client.Checked,'on'))
    end
end

function movieException = start_processing_movies_in_parallel_parfeval(movieRun,handles,movieException, procRun)
    
    currentPool = gcp('nocreate');
    poolSize = findobj(handles.menu_parallel_pool,'Checked','on');
    poolSize = str2double(poolSize.Label);
    if(~isempty(currentPool))
        if(currentPool.NumWorkers ~= poolSize ...
                || ~strcmp(get(uTrackParCluster,'Profile'), ...
                   currentPool.Cluster.Profile))
            delete(currentPool);
            currentPool = [];
        end
    end
    if(isempty(currentPool))
        currentPool = parpool(uTrackParCluster,poolSize);
    end

    for i=1:length(movieRun)
        iMovie = movieRun(i);

        userData = selectMovie(iMovie, handles, procRun);
        
        % Determine sequence that parent processes and checked processes
        % should be run
        procSequence = userData.crtPackage.getProcessSequence(procRun{iMovie});
        % Determine which processes were successful
        procSuccess =  cellfun(@(proc) proc.success_,userData.crtPackage.processes_(procSequence));
        % Only run the processes that were either requested or were not
        % successful
        mustRun = ismember(procSequence,procRun{iMovie});
        procSequence = procSequence(~procSuccess | mustRun);
        procs = userData.crtPackage.processes_(procSequence);
        % Save procSequence for reload
        procRun{iMovie} = procSequence;
%         f = @(p,~) cellfun(@run,p);
        jobs(i) = parfeval(currentPool,@runProcesses,1,procs,0);
        set(handles.text_status, 'String', ...
            sprintf('Queuing %d of %d movies total ...', i, length(movieRun)));
    end
    set(handles.text_status, 'String', ...
        sprintf('Waiting for parallel job %d of %d movies total ...', 0, length(movieRun)));
    
    % TODO: Do something better with this
    finishedJobs = zeros(size(jobs));
    finishedJobCount = 0;
    while(finishedJobCount ~= length(jobs))
        for i=1:length(jobs)
            if(~strcmp(jobs(i).State,'pending'))
                if(~finishedJobs(i))
                    if(wait(jobs(i),'finished',1))
                        finishedJobs(i) = true;
                        finishedJobCount = sum(finishedJobs);
                        
                        % Update Processes
                        newProcs = fetchOutputs(jobs(i));
                        iMovie = movieRun(i);
                        userData = updateUserData(handles, newProcs(1), iMovie);
                        
                        set(handles.text_status, 'String', ...
                             sprintf('Finished parallel job %d of %d',finishedJobCount,length(jobs)));
                        fprintf('Movie %d output:\n',movieRun(i));
                        disp(jobs(i).Diary);
                        fprintf('\n');
                        packageGUI_RefreshFcn(handles,'initialize');
                    elseif(strcmp(jobs(i).State,'failed'))
                        finishedJobs(i) = true;
                        finishedJobCount = sum(finishedJobs);
                        % Save the error into movie Exception cell array
                        ME = jobs(i).Error;
                        ME2 = MException('lccb:run:error','Parallel Run: Movie %d',...
                            i);
                        movieException{iMovie} = horzcat(movieException{iMovie}, ME2);
                        movieException{iMovie}=movieException{iMovie}.addCause(ME);

                        procRun{iMovie} = procRun{iMovie}(procRun{iMovie} < procID);

                        % Refresh wall status
                        packageGUI_RefreshFcn(handles,'initialize');
                    end
                end
            end
        end
    end
    for i=1:length(jobs)
        delete(jobs(i));
    end
end

function movieException = start_processing_movies_in_parallel_single_batch(movieRun,handles,movieException, procRun)
    % Process all movies in a single batch job
    
    poolSize = findobj(handles.menu_parallel_pool,'Checked','on');
    poolSize = str2double(poolSize.Label);
    poolParams = {};
    if(~isnan(poolSize))
        poolParams = {'Pool',poolSize};
    end
    
    procsPerMovie = cell(size(movieRun));
    
    for i=1:length(movieRun)
        iMovie = movieRun(i);
        
        userData = selectMovie(iMovie, handles, procRun);
        
        % Determine sequence that parent processes and checked processes
        % should be run
        procSequence = userData.crtPackage.getProcessSequence(procRun{iMovie});
        % Determine which processes were successful
        procSuccess =  cellfun(@(proc) proc.success_,userData.crtPackage.processes_(procSequence));
        % Only run the processes that were either requested or were not
        % successful
        mustRun = ismember(procSequence,procRun{iMovie});
        procSequence = procSequence(~procSuccess | mustRun);
        procs = userData.crtPackage.processes_(procSequence);
        procsPerMovie{i} = procs;
    end

    
    job = batch(uTrackParCluster,@start_processing_movies_in_parallel_single_batch_job,2,{movieRun,procsPerMovie,movieException},'AutoAttachFiles',false,'CaptureDiary',true,poolParams{:});
    disp(job);
    wait(job);
    out = fetchOutputs(job);
    movieException = out{1};
    
    % Update Processes
    procs = out{2};
    for i=1:length(movieRun)
        % Update Processes
        iMovie = movieRun(i);
        userData = updateUserData(handles, procs{i}(1), iMovie);
    end
    
    job.diary;
    delete(job);
    
    % Refresh wall status
    packageGUI_RefreshFcn(handles,'initialize');

end

function [movieException,procs] = start_processing_movies_in_parallel_single_batch_job(movieRun,procsPerMovie,movieException)
    currentPool = gcp('nocreate');
    procs = cell(1,length(movieRun));
    if(isempty(currentPool))
        % No parallel pool available, run in serial
        for i=1:length(movieRun)
            iMovie = movieRun{i};
                       
            fprintf('Movie %d\n',movieRun(i));
            try
                procs{i} = runProcesses(procsPerMovie{i});
            catch ME
                ME2 = MException('lccb:run:error','Parallel Run: Movie %d',...
                    i);
                movieException{iMovie} = horzcat(movieException{iMovie}, ME2);
                movieException{iMovie}=movieException{iMovie}.addCause(ME);

%                 procRun{iMovie} = procRun{iMovie}(procRun{iMovie} < procID);
            end
        end
    else
        % Parallel pool available, use it!
        
        % Simpler parfor version
%         exceptions = movieException(movieRun);
%         parfor i=1:length(movieRun)
% %             iMovie = movieRun{i};
%             fprintf('Movie %d\n',movieRun(i));
%             try
%                 runProcesses(procsPerMovie{i});
%             catch ME
%                 ME2 = MException('lccb:run:error','Parallel Run: Movie %d',...
%                     i);
%                 exceptions{i} = horzcat(exceptions{i}, ME2);
%                 exceptions{i}= exceptions{i}.addCause(ME);
% 
% %                 procRun{iMovie} = procRun{iMovie}(procRun{iMovie} < procID);
%             end
%         end
%         movieException(movieRun) = exceptions;
        
        % Copied from pool version
        % TODO: Unify code
        currentPool = gcp;
        for i=1:length(movieRun)
            iMovie = movieRun(i);
            jobs(i) = parfeval(currentPool,@runProcesses,1,procsPerMovie{i},0);
%             set(handles.text_status, 'String', ...
%                 sprintf('Queuing %d of %d movies total ...', i, length(movieRun)));
        fprintf('Queuing %d of %d movies total ...\n', i, length(movieRun));
        end
%         set(handles.text_status, 'String', ...
%             sprintf('Waiting for parallel job %d of %d movies total ...', 0, length(movieRun)));
        fprintf('Waiting for parallel job %d of %d movies total ...\n', 0, length(movieRun));

        % TODO: Do something better with this
        finishedJobs = zeros(size(jobs));
        finishedJobCount = 0;
        while(finishedJobCount ~= length(jobs))
            for i=1:length(jobs)
                if(~strcmp(jobs(i).State,'pending'))
                    if(~finishedJobs(i))
                        if(wait(jobs(i),'finished',1))
                            finishedJobs(i) = true;
                            finishedJobCount = sum(finishedJobs);
                            
                            % Fetch proceses to update later
                            newProcs = fetchOutputs(jobs(i));
                            procs{i} = newProcs{1};
                            
%                             set(handles.text_status, 'String', ...
%                                  sprintf('Finished parallel job %d of %d',finishedJobCount,length(jobs)));
                            fprintf('Finished parallel job %d of %d\n',finishedJobCount,length(jobs))
                            fprintf('Movie %d output:\n',movieRun(i));
                            disp(jobs(i).Diary);
                            fprintf('\n');
%                             packageGUI_RefreshFcn(handles,'initialize');
                        elseif(strcmp(jobs(i).State,'failed'))
                            finishedJobs(i) = true;
                            finishedJobCount = sum(finishedJobs);
                            % Save the error into movie Exception cell array
                            ME = jobs(i).Error;
                            ME2 = MException('lccb:run:error','Parallel Run: Movie %d',...
                                i);
                            movieException{iMovie} = horzcat(movieException{iMovie}, ME2);
                            movieException{iMovie}=movieException{iMovie}.addCause(ME);

%                             procRun{iMovie} = procRun{iMovie}(procRun{iMovie} < procID);

                            % Refresh wall status
%                             packageGUI_RefreshFcn(handles,'initialize');
                        end
                    end
                end
            end
        end
        for i=1:length(jobs)
            delete(jobs(i));
        end

    end
end

function movieException = start_processing_movies_in_parallel_batch_per_movie(movieRun,handles,movieException, procRun)
    % Process each movie in a separate batch job
    
    poolSize = findobj(handles.menu_parallel_pool,'Checked','on');
    poolSize = str2double(poolSize.Label);
    poolParams = {};
    if(~isnan(poolSize))
        poolParams = {'Pool',poolSize};
    end

    for i=1:length(movieRun)
        iMovie = movieRun(i);
        
        userData = selectMovie(iMovie, handles, procRun);
        
        % Determine sequence that parent processes and checked processes
        % should be run
        procSequence = userData.crtPackage.getProcessSequence(procRun{iMovie});
        % Determine which processes were successful
        procSuccess =  cellfun(@(proc) proc.success_,userData.crtPackage.processes_(procSequence));
        % Only run the processes that were either requested or were not
        % successful
        mustRun = ismember(procSequence,procRun{iMovie});
        procSequence = procSequence(~procSuccess | mustRun);
        procs = userData.crtPackage.processes_(procSequence);
%         f = @(p,~) cellfun(@run,p);
%         jobs(i) = parfeval(gcp,f,0,procs,0);
        jobs(i) = batch(uTrackParCluster,@runProcesses,1,{procs,0},'AutoAttachFiles',false,'CaptureDiary',true,poolParams{:});
        procRun{iMovie} = procSequence;

        set(handles.text_status, 'String', ...
            sprintf('Queuing %d of %d movies total ...', i, length(movieRun)));
    end

    set(handles.text_status, 'String', ...
        sprintf('Waiting for parallel job %d of %d movies total ...', 0, length(movieRun)));
    
    % TODO: Do something better with this
    finishedJobs = zeros(size(jobs));
    finishedJobCount = 0;
    while(finishedJobCount ~= length(jobs))
        for i=1:length(jobs)
            if(~strcmp(jobs(i).State,'pending'))
                if(~finishedJobs(i))
                    if(wait(jobs(i),'finished',1))
                        finishedJobs(i) = true;
                        finishedJobCount = sum(finishedJobs);
                        
                        % Update Processes
                        newProcs = fetchOutputs(jobs(i));
                        newProcs = newProcs{1};
                        iMovie = movieRun(i);
                        userData = selectMovie(iMovie, handles, procRun);
                        userData = updateUserData(handles, newProcs(1), iMovie);
                        
                        set(handles.text_status, 'String', ...
                             sprintf('Finished parallel job %d of %d',finishedJobCount,length(jobs)));
                        fprintf('Movie %d output:\n',movieRun(i));
                        jobs(i).diary;
                        fprintf('\n');
                        packageGUI_RefreshFcn(handles,'initialize');
                    elseif(strcmp(jobs(i).State,'failed'))
                        finishedJobs(i) = true;
                        finishedJobCount = sum(finishedJobs);
                        % Save the error into movie Exception cell array
                        ME = jobs(i).Error;
                        ME2 = MException('lccb:run:error','Parallel Run: Movie %d',...
                            i);
                        movieException{iMovie} = horzcat(movieException{iMovie}, ME2);
                        movieException{iMovie}=movieException{iMovie}.addCause(ME);

                        procRun{iMovie} = procRun{iMovie}(procRun{iMovie} < procID);

                        % Refresh wall status
                        packageGUI_RefreshFcn(handles,'initialize');
                    end
                end
            end
        end
    end
    for i=1:length(jobs)
        delete(jobs(i));
    end
end


function userfcn_runProc_dfs (procID, procRun, handles)  % throws exception

% Set user Data
userData = get(handles.figure1, 'UserData');

parentRun = [];
parentID=userData.crtPackage.getParent(procID);

% if current process procID have dependency processes    
for j = parentID
    % if parent process is one of the processes need to be run
    % if parent process has already run successfully
    if any(j == procRun) && ~userData.crtPackage.processes_{j}.success_
        parentRun = horzcat(parentRun,j); %#ok<AGROW>
    end
end

% if above assumptions are yes, recursively run parent process' dfs fcn
for j = parentRun
    userfcn_runProc_dfs (j, procRun, handles)
end

try
    userData.crtPackage.processes_{procID}.run(); % throws exception
catch ME
    rethrow(ME)
end

% Refresh wall status
packageGUI_RefreshFcn(handles,'initialize');
end

function userData = selectMovie(iMovie,handles,procRun)
% selectMovie Selects the movie of interest, updating GUI and userData
% Importantly also sets userData.crtPackage via switchMovie_Callback
    userData = get(handles.figure1, 'UserData');

    if iMovie ~= userData.id
        % Update the movie pop-up menu in the main package GUI
        set(handles.figure1, 'UserData', userData)
        set(handles.popupmenu_movie, 'Value', iMovie)
        
        % Update the movie pop-up menu in the main package GUI
        packageGUI('switchMovie_Callback',handles.popupmenu_movie, [], handles)
        userData = get(handles.figure1, 'UserData');
    end
    
    % Clear icons of selected processes
    % Return user data !!!
    set(handles.figure1, 'UserData', userData)
    userfcn_drawIcon(handles,'clear',procRun{iMovie},'',true); % user data is retrieved, updated and submitted
    % Refresh user data !!!
    userData = get(handles.figure1, 'UserData');
    set(handles.text_status, 'Visible', 'on');
end

function userData = updateUserData(handles,newProcs,iMovie)
%updateUserData update userData to replace all altered MovieData and Package handles
% If Process is run remotely, local handles may not be updated. The only way to update
% all these handles is to alter userData
    assert(length(newProcs) == length(iMovie), ...
        'packageGUI_RunFcn:updateUserData:argLength', ...
        'Arguments must be of the same length');
    userData = get(handles.figure1,'UserData');

    if(isa(newProcs,'Process'))
        newProcs = {newProcs};
    end

    % Determine field
    if ~isempty(userData.MD), field='MD'; else field = 'ML'; end

    for i=1:length(newProcs)
        proc = newProcs{i};

        % Replace MovieObject
        MOs = userData.(field);
        assert(strcmp(MOs(iMovie(i)).getFullPath, ...
                      proc.getOwner().getFullPath), ...
            'packageGUI_RunFcn:updateUserData:movieObjectFullPathDiffers', ...
            'Attempt to replace MovieObject with one with a different path');
        MOs(iMovie(i)) = proc.getOwner();
        userData.(field) = MOs;
        
        if(strcmp(field,'MD'))
            % Update MovieList(s) if we updated the MD
            % Loop because there may be more than one MovieList
            for ll = 1:length(userData.ML)
                % attachMovies only attaches if the movie is already in the
                % list and does not generate an error otherwise
                userData.ML(ll).attachMovies(MOs(iMovie(i)));
            end
        end

        % Replace Package
        iPackage = MOs(iMovie(i)).getPackageIndex(userData.packageName,1,false);
        userData.package(iMovie(i)) = userData.MD(iMovie).getPackage(iPackage);

        % Update crtPackage if it is the one currently selected
        if(userData.id == iMovie)
            userData.crtPackage = userData.package(iMovie(i));
        end
    end

    % Update userData
    set(handles.figure1, 'UserData', userData)
end
