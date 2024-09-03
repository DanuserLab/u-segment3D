classdef MatFileCache < cached.FileCache
    %MatFileCache implements a caching mechanism for Matlab MAT files
    %
    % This allows for a separate cache instances rather than a global cache as in cached.load and cached.save
    %
    % See also cached.load, cached.save 
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
    
    properties
        % if true, then the cache is only valid if the modification
        % timestamp and file size are exactly the same
        checkValidity
        % save variables to disk when one of the save methods is invoked
        saveToDisk
    end
    
    methods
        function obj = MatFileCache(varargin)
            %MatFileCache takes two parameters corresponding to the class
            %properties
            %
            % checkValidity
            % saveToDisk
            obj = obj@cached.FileCache(@defaultLoad);
            ip = inputParser;
            ip.addParameter('checkValidity',true,@islogical);
            ip.addParameter('saveToDisk',true,@islogical);
            ip.parse(varargin{:});
            obj.checkValidity = ip.Results.checkValidity;
            obj.saveToDisk = ip.Results.saveToDisk;
        end
        function saveStruct(obj,fileName,S,varargin)
            % saveStruct(fileName,S, ...) implements saving as a struct
            key = obj.resolveKey(fileName);
            R.all_variables = true;
            obj.setData(key,S,R);
            if(obj.saveToDisk)
                save(key,'-struct','S',varargin{:});
            end
        end
        function saveVariables(obj,fileName,varargin)
            % saveVariables(fileName, variables ...)
            varList = strjoin(varargin,',');
            [values{1:length(varargin)}] = evalin('caller',['deal(' varList ');']);
            S = cell2struct(values(:),varargin(:));
            obj.saveStruct(fileName,S);
        end
        function save(obj,varargin)
            % save is similar to the builtin save. -append and -regexp are currently not supported.
            % Use saveStruct instead of the -struct option
            % fileName is the first argument
            key = obj.resolveKey(varargin{1});
            
            isOption = strncmp(varargin,'-',1);
            options = varargin(isOption);
            appendFlag = any(strcmp(options,'-append'));
            regexpFlag = any(strcmp(options,'-regexp'));
            structFlag = any(strcmp(options,'-struct'));

            if(appendFlag)
                oldS = obj.load(key);
            end
            
            % call the builtin save function in the caller workspace
            if(obj.saveToDisk)
                evalin('caller',['save(''' strjoin(varargin,''',''') ''')']);
            end
                                     
            % extract variables from caller (see saveVariables)
            variables = varargin(~isOption);
            % do not include filename in variables list
            variables = variables(2:end);

            % Goal: Build a struct containing the specified variables and pass to saveStruct

            if(structFlag)
                % get the struct from caller
                S = evalin('caller',variables{1});
                if(regexpFlag)
                    % if struct and regexp, filter the fields
                    S = regexpfields(S,variables{:});
                end
            elseif(regexpFlag)
                % if not struct and regexp, filter the caller workspace
                argList = strjoin(['caller' '-regexp' variables],''',''');
                S = evalin('caller',['workspace2struct(''' argList ''')']);
            elseif(isempty(variables))
                S = evalin('caller','worksapce2struct');
            else
                argList = strjoin(['caller' variables],''',''');
                S = evalin('caller',['workspace2struct(''' argList ''')']);
            end

            if(appendFlag)
                % if appendFlag, then merge the old struct and the new struct
                % new fields take precedence
                S = merge(S,oldS);
            end

            
            % save in cache, but not to disk since we already did that
            R.all_variables = true;
            obj.setData(key,S,R);
        end
        function [S , cached] = load(obj,fileName,varargin)
            % MatFileCache.load is similar to the built-in load. All options are ignored.
            key = obj.resolveKey(fileName);

            [S,R] = obj.retrieve(key);
            
            isOption = strncmp(varargin,'-',1);
            variables = varargin(~isOption);
            if(~isempty(variables))
                if(any(~isfield(S,variables)))
                    % if any of the requested variables are not present,
                    % then update the cache
                    S_fields = fieldnames(S);
                    newVariables = setdiff(variables,S_fields);
                    newS = load(filename,newVariables{:});
                    cached = false;
                    newS_fields = fieldnames(newS);
                    % NB: Documented tip in struct2cell:
                    % Order is preserved between fieldnames and struct2cell
                    S = cell2struct([ struct2cell(S) ; struct2cell(newS)], ...
                                    [ S_fields       ; newS_fields]      , 1);
                    % all variables were not loaded
                    R.all_variables = false;
                    obj.setData(key,S,R);
                end
                
                % Limit output structure to only the variables requested
                bigS = S;
                S = struct();
                for f = 1:length(variables)
                    S.(variables{f}) = bigS.(variables{f});
                end
                
            elseif(~R.all_variables)
                % no variables specified and all variables were not cached
                S = load(key);
                R.all_variables = true;
                obj.setData(key,S,R);
                cached = false;
            end

            % If no output is requested, assign values in the caller like builtin
            if(nargout == 0)
                % if we loaded in ascii mode, then S may not be a struct
                if(isstruct(S))
                    S_fields = fieldnames(S);
                    for f = 1:length(S_fields)
                        assignin('caller',S_fields{f},S.(S_fields{f}));
                    end
                end
            end
        end
    end
    methods ( Access = protected )
        function key = resolveKey(obj,fileName)
            if(obj.cache.isKey(fileName))
                key = fileName;
            else
                key = whichMatFile(fileName);
            end
        end
        function valid_tf =  isValid_(obj,key)
            if(~obj.checkValidity)
                valid_tf = true;
            else
                valid_tf = obj.isValid_@cached.FileCache(key);
            end
        end
        function record = setData(obj,key,data,R)
            % if no additional record is specified, assume all variables have been loaded
            if(nargin < 4)
                R = struct('all_variables',true);
            end
            record = obj.setData@cached.FileCache(key,data,R);
        end
    end
end
function [S, record] = defaultLoad(fileName)
    % by default, load the entire file with all variables
    S = load(fileName);
    record.all_variables = true;
end

