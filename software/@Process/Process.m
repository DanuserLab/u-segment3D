classdef Process < hgsetget
    % Defines the abstract class Process from which every user-defined process
    % will inherit.
    %
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
    
    properties (SetAccess = private, GetAccess = public)
        owner_          % Movie data object owning the process
        createTime_     % Time process was created
        startTime_      % Time process was last started
        finishTime_     % Time process was last run
    end
    
    properties  (SetAccess = protected)
        name_           % Process name
        tag_            % User definable/queriable tag for orgnaizational convenience
        processTree_    % Previous proceses associated with MovieData owner at time of process instantiation
        
        % Success/Uptodate flags
        procChanged_   % Whether process parameters have been changed
        success_       % If the process has been successfully run
        % If the parameter of parent process is changed
        % updated_ - false, not changed updated_ - true
        updated_

        funName_        % Function running the process
        funParams_      % Parameters for running the process
        
        inFilePaths_    % Path to the process input
        outFilePaths_   % Path to the process output
        outFilePathsCache_;

        is3Dcompatible_ % can process handle 3D movie data

    end
    properties
        notes_          % Process notes
    end
    properties (Transient=true)
        displayMethod_  % Cell array of display methods
    end
    methods (Access = protected)
        function obj = Process(owner, name)
            % Constructor of class Process
            if nargin > 0
                %Make sure the owner is a MovieData object
                if isa(owner,'MovieObject')
                    obj.owner_ = owner;
                else
                    error('lccb:Process:Constructor','The owner of a process must always be a movie object!')
                end
                
                if nargin > 1
                    obj.name_ = name;
                end

                obj.createTime_ = clock;
                obj.procChanged_ = false;
                obj.success_ = false;
                obj.updated_ = true;
                obj.processTree_ = owner.processes_;
                
                % set default tag label
                numproc = numel(owner.getProcessIndex(class(obj), Inf, 0));
                obj.tag_ = [class(obj) '_' num2str(numproc+1)];
                obj.outFilePathsCache_=[];
            end
        end
    end
    
    methods
        
        function owner = getOwner(obj)
            % Retrieve the process owner
            owner = obj.owner_;
        end
        
        function parameters = getParameters(obj)
            % Get the process parameters
            parameters = obj.funParams_;
        end

        function is3Dcompatible = get3DCompatible(obj)
            % Get the process parameters
            is3Dcompatible = obj.is3Dcompatible_;
        end
        
        function setParameters(obj, para,varargin)
            % If the arguments are a field,value pair list
            % Then update the current parameters with those values
            if(ischar(para) && mod(nargin,2) == 1)
                in = [para varargin];
                para = obj.funParams_;
                for ii = 1:2:length(in)
                    para.(in{ii}) = in{ii+1};
                end
            end
            % Set the process parameters
            if isequal(obj.funParams_, para), return; end
            obj.funParams_ = para;
            obj.procChanged_= true;
            
            % Run sanityCheck on parent package to update dependencies
            for packId = obj.getPackageIndex()
                obj.getOwner().getPackage(packId).sanityCheck(false,'all');
            end
        end
        
        function setPara(obj, parameters, varargin)
            setParameters(obj, parameters, varargin{:})
        end
        
        function setUpdated(obj, is)
            % Set update status of the current process
            % updated - true; outdated - false
            obj.updated_ = is;
        end
        
        function setDateTime(obj)
            %The process has been re-run, update the time.
            obj.finishTime_ = clock;
        end
        
        function status = checkChanNum(obj,iChan)
            assert(~isempty(iChan) && isnumeric(iChan),'Please provide a valid channel input');
            status = insequence(iChan, 1,numel(obj.getOwner().channels_));
        end
        
        function status = checkFrameNum(obj,iFrame)
            assert(~isempty(iFrame) && isnumeric(iFrame),'Please provide a valid frame input');
            status = insequence(iFrame, 1, obj.getOwner().nFrames_);
        end
        
        function status = checkDepthNum(obj, iZ)
            assert(~isempty(iZ) && isnumeric(iZ),'Please provide a valid z input');
            status = insequence(iZ, 1,obj.getOwner().zSize_);
        end
        
        function sanityCheck(obj)
            % Perform saniy check on the process
            
            % Ensure obj.funName_ is a valid function handle
            if(~isa(obj.funName_,'function_handle'))
                % If funName_ is a char, convert to function_handle
                if(ischar(obj.funName_))
                    obj.funName_ = str2func(obj.funName_);
                else
                    error('lccb:Process:sanityCheck:funNameType', ...
                        '%s is not a char or function_handle.', ...
                        func2str(obj.funName_));
                end
            end
            
            % Make sure funName_ exists and takes at least one input
            try
                funNameIn = nargin(obj.funName_);
                if(funNameIn == 0)
                    error('lccb:Process:sanityCheck:funNameInput', ...
                        '%s does not take a MovieObject argument.', ...
                        func2str(obj.funName_));
                end
            catch err
                if(strcmp(err.identifier,'MATLAB:narginout:notValidMfile'))
                    % funName_ does not exist
                    error('lccb:Process:sanityCheck:funNameDoesNotExist', ...
                        '%s does not exist.',func2str(obj.funName_));
                else
                    % funName_ exists but there's another problem
                    rethrow(err);
                end
            end

            % Retrieve current process parameters and default parameters
            crtParams = obj.getParameters();
            % Interpret empty as a struct with no fields
            if(isempty(crtParams))
                crtParams = struct();
            end
            if(~isstruct(crtParams))
                error('lccb:Process:sanityCheck:parametersMustBeStruct', ...
                    'Process parameters must be a struct or empty.');
            end
            defaultParams = obj.getDefaultParams(obj.getOwner());
            if(isempty(defaultParams))
                defaultParams = struct();
            end
            if(~isstruct(defaultParams))
                error('lccb:Process:sanityCheck:defaultParametersMustBeStruct', ...
                    'Default parameters must be a struct or empty.');
            end
            crtFields = fieldnames(crtParams);
            defaultFields = fieldnames(defaultParams);
            
            %  Find undefined default parameters
            status = ismember(defaultFields, crtFields);
            if all(status), return; end
            
            % Add default missing fields and set process parameters
            missingFields = defaultFields(~status);
            for i = 1: numel(missingFields)
                missingField = missingFields{i};
                % Permit struct arrays with missing fields
                [crtParams.(missingField)] = deal(defaultParams.(missingField));
            end
            obj.setParameters(crtParams);
        end
        
        function new = copy(this)
            % Instantiate new object of the same class.
            new=feval(class(this),this.getOwner(),this.name_);
            % Copy all non-hidden properties.
            p = properties(this);
            for i = 1:length(p)
                if(~strcmp(p{i},'owner_')&&~strcmp(p{i},'createTime_')&&~strcmp(p{i},'startTime_')&&~strcmp(p{i},'finishTime_'))
                    new.(p{i}) = this.(p{i});
                end
            end
        end
        
        function run(obj,varargin)
            % Reset sucess flags and existing display methods
            obj.resetDisplayMethod();
            obj.success_=false;
            
            % Run the process!
            obj.startTime_ = clock;

            if(isa(obj,'NonSingularProcess'))
                % Run by passing the Process itself rather than
                % owner which can be obtained easily with getOwner()
                % New style
                obj.funName_(obj, varargin{:});
            else
                % Run by passing the Process owner,
                % Usually a MovieObject like MovieData
                % Legacy style
                obj.funName_(obj.getOwner(), varargin{:});
            end

            % Update flags and set finishTime
            obj.success_= true;
            obj.updated_= true;
            obj.procChanged_= false;
            obj.finishTime_ = clock;
            
            % Run sanityCheck on parent package to update dependencies
            for packId = obj.getPackageIndex()
                obj.getOwner().getPackage(packId).sanityCheck(false,'all');
            end
            
            obj.getOwner().save();
        end
        
        function resetDisplayMethod(obj)
            if isempty(obj.displayMethod_), return; end
            validMethods = ~cellfun(@isempty,obj.displayMethod_);
            cellfun(@delete, obj.displayMethod_(validMethods));
            obj.displayMethod_ = {};
        end
       
       function outputCell=loadFileOrCache(obj)
           % if(isempty(obj.outFilePathsCache_))
                outputCell=cell(1,numel(obj.outFilePaths_));
                for i=1:numel(obj.outFilePaths_)
                    if(~isempty(obj.outFilePaths_{i}))
                        outputCell{i}=load(obj.outFilePaths_{i});
                    end
                end
            % else
            %     outputCell=obj.outFilePathsCache_;
            % end
       end

       % function cacheProcess=createCacheProcess(obj)
       %      cacheProcess=ExternalProcess(obj.getOwner(),['cached_' obj.name_]);
       %      cacheProcess.setOutFilePaths({[obj.getOwner().outputDirectory_ filesep 'dynROIs' filesep 'initDynROIs.mat']});
       %      cacheProcess.setProcessTag(['cached_' obj.tag_]);
       %      eoutput=cellfun(@isempty,obj.outFilePaths_);
       %      tmp=cell(size(obj.outFilePaths_));
       %      tmp(~eoutput)=cellfun(@(f) load(f),obj.outFilePaths_(~eoutput),'unif',0);
       %      cacheProcess.outFilePathsCache_=tmp;
       % end


       function emptyCache(obj)
                obj.outFilePathsCache_=[];
       end

       
        
        
        function method = getDisplayMethod(obj, iOutput, iChan)
            
            if any(size(obj.displayMethod_) < [iOutput iChan])
                method = [];
            else
                method = obj.displayMethod_{iOutput,iChan};
            end
        end
        
        function setDisplayMethod(obj,iOutput,iChan,displayMethod)
            
            assert(isa(displayMethod(), 'MovieDataDisplay'));
            if ~isempty(obj.getDisplayMethod(iOutput, iChan))
                delete(obj.getDisplayMethod(iOutput, iChan));
            end
            obj.displayMethod_{iOutput,iChan} = displayMethod;
        end
        
        
        function setInFilePaths(obj,paths)
            %  Set input file paths
            obj.inFilePaths_=paths;
        end
        
        function setOutFilePaths(obj,paths,chanNum)
            % Set output file paths
%             obj.outFilePaths_ = paths;
            if nargin <3
                obj.outFilePaths_ = paths;
            else
                if obj.checkChanNum(chanNum)
                    obj.outFilePaths_{1,chanNum} = paths;
                else
                    error('lccb:set:fatal','Invalid channel number for detection path!\n\n');
                end                
            end
             

        end

        function setProcessTag(obj, tag)
            obj.tag_ = tag;
        end
        function setFunName(obj, funName)
            obj.funName_ = funName;
        end
        function tag = getProcessTag(obj)
            if isprop(obj, 'tag_')
                tag = obj.tag_;
            else
                numproc = numel(obj.owner_.getProcessIndex(class(obj), Inf, 0));
                tag = ['proc_' class(obj) '_' num2str(numproc+1)];
                obj.setProcessTag(tag);
            end
        end

        function time = getProcessingTime(obj)
            %The process has been re-run, update the time.
            time=sec2struct(24*3600*(datenum(obj.finishTime_)-datenum(obj.startTime_)));
        end
        
        function [packageID, procID] = getPackageIndex(obj)
            % Retrieve index of packages to which the process is associated
            validPackage = cellfun(@(x) (~isempty(x))&&(x.hasProcess(obj)) ,...
                obj.getOwner().packages_);
            packageID = find(validPackage);
            procID = cellfun(@(x) x.getProcessIndex(obj),...
                obj.getOwner().packages_(validPackage));
        end
        
        function relocate(obj,oldRootDir,newRootDir)
            % Relocate all paths in various process fields
            relocateFields ={'inFilePaths_','outFilePaths_', 'funParams_'};
            for i=1:numel(relocateFields)
                obj.(relocateFields{i}) = relocatePath(obj.(relocateFields{i}),...
                    oldRootDir,newRootDir);
                
                % special relocate for BuildDynROIProcess in NewUTrack3DPackage and its dynROI_rawImages MD
                if isa(obj, 'BuildDynROIProcess') && isequal(relocateFields{i}, 'outFilePaths_')
                    load(obj.outFilePaths_{3, 1});
                    movieDataDynROICell{1} = relocatePath(movieDataDynROICell{1}, oldRootDir,newRootDir);
                    save(obj.outFilePaths_{3, 1},'movieDataDynROICell');
                    clear movieDataDynROICell
                    
                    % relocate for MD built based on DynROI raw images.
                    s = cached.load(obj.outFilePaths_{3, 1}, '-useCache', true);
                    load(s.movieDataDynROICell{1});
                    MD.relocate(oldRootDir,newRootDir, true);
                    MD.save
                    clear s MD
                end    
            end
        end
        
        function hfigure = resultDisplay(obj)
            if isa(obj.getOwner(), 'ImageData')
                hfigure = imageDataViewer(obj.getOwner(), ...
                    find(cellfun(@(x)isequal(x,obj),obj.getOwner().processes_)));
            else
            hfigure = movieViewer(obj.getOwner(), ...
                find(cellfun(@(x)isequal(x,obj),obj.getOwner().processes_)));
            end
        end
        
        function h=draw(obj,iChan,varargin)
            % Template function to draw process output
            
            % Input check
            if ~ismember('getDrawableOutput',methods(obj)), h=[]; return; end
            outputList = obj.getDrawableOutput();
            ip = inputParser;
            ip.addRequired('obj',@(x) isa(x,'Process'));
            ip.addRequired('iChan',@isnumeric);
            ip.addOptional('iFrame',[],@isnumeric);
            ip.addParameter('output',outputList(1).var,@(x) any(cellfun(@(y) isequal(x,y),{outputList.var})));
            ip.addParameter('useCache', false, @islogical);
            ip.addParameter('movieOverlay', false, @islogical);
            ip.KeepUnmatched = true;
            if obj.owner_.is3D()
                ip.addOptional('iZ',[],@(x) insequence(x,1,obj.owner_.zSize_));
                ip.addOptional('projectionAxis3D','Z', @(x) ismember(x,{'X','Y','Z','three'}));
            end
            ip.parse(obj,iChan,varargin{:});
            
            % Load data
            loadArgs{1} = iChan;
            if ~isempty(ip.Results.iFrame)
                loadArgs{2} = ip.Results.iFrame;
            end

            % use inputParser structExpand
            loadParams.output = ip.Results.output;
            if obj.owner_.is3D()
                % could owner be 3D without having an iFrame ?
                loadParams.iZ = ip.Results.iZ;
                loadParams.projectionAxis3D = ip.Results.projectionAxis3D;
            end
            if ip.Results.useCache
                loadParams.useCache = true;
            end
            loadArgs = [loadArgs loadParams];
            
            data = obj.loadChannelOutput(loadArgs{:});
                
            iOutput= find(cellfun(@(y) isequal(ip.Results.output,y),{outputList.var}));
            if ~isempty(outputList(iOutput).formatData),
                data=outputList(iOutput).formatData(data);
            end
            
            % Initialize display method
            if isempty(obj.getDisplayMethod(iOutput,iChan))
                obj.setDisplayMethod(iOutput, iChan,...
                    outputList(iOutput).defaultDisplayMethod(iChan));
            end
            
            % Create graphic tag and delegate drawing to the display class
            tag = ['process' num2str(obj.getIndex()) '_channel' num2str(iChan) '_output' num2str(iOutput)];
            if ip.Results.movieOverlay % if not channel specific (to match what movieViewer expects)
                tag = ['process' num2str(obj.getIndex()) '_output' num2str(iOutput)];
            end
            h = obj.getDisplayMethod(iOutput, iChan).draw(data,tag,ip.Unmatched);
        end
        
        function index = getIndex(obj)
            % Retrieve index of process in the owner object
            index = find(cellfun(@(x) isequal(x,obj),obj.getOwner().processes_));
            assert(numel(index)==1);
        end
        
        [ movieObject, process, processID ] = getOwnerAndProcess( process, processClass, createProcessIfNoneExists, varargin );
    end
    
    methods (Static)
        function status = isProcess(name)
            % Check if the input classname is of class Process
            status = exist(name, 'class') == 8 && isSubclass(name, 'Process');
        end
        
        function status = hasGUI(name)
            % Check if process has a settings graphical interface
            m = meta.class.fromName(name);
            status = ismember('GUI',{m.MethodList.Name});
        end
        
    end
    
    methods (Static,Abstract)
        getDefaultParams
        getName
    end
end
