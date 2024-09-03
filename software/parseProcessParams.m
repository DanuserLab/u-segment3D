function [paramOut,unrecParam] = parseProcessParams(procOb,paramIn,asCell)
%PARSEPROCESSPARAMS parses the parameters of the input process object
% 
% [paramOut] = parseProcessParams(procObj,paramIn)
% [paramOut,unrecParam] = parseProcessParams(procObj,paramIn)
% [paramOut,unrecParam] = parseProcessParams(procObj,paramIn,asCell)
% 
% This function returns the individual parameters that are specified in the
% input process object's field "funParams" If the input parameter structure
% "paramIn" is specified, parameters specified in this structure will
% override any values in procOb.funParams_. They will be stored in the
% procOb and returned as outputs.
% 
% Additionally, the function can also return a structure specifying the
% input parameters which were input but were NOT in the process object.
%
% Input:
%   
%   procOb - An object of the class Process.m
% 
%   paramIn - A structure or object containing fields whose name's match
%   with the elements of paramList and who'se values are to be used as
%   parameters, replacing those stored in procOb. Optional. If not input,
%   all the values in procOb.funParams_ will be returned instead.
%
%   asCell - If true, and the unrecognized parameters are requested, the
%   unregonized parameters will be returned as a cell-array instead of a
%   structure, in the format:
%
%   {'FieldName1',fieldValue1,'FieldName2',fieldValue2,...
%
%   This format may be useful for passing unrecognized input fields to
%   another function which uses varargin. If false, the unrecognized
%   parameters will be returned as a structure.
%   Optional. Default is False.
%
% Output:
%
%   paramOut - A structure containing the new parameters. Additionally the
%   parameters will be updated in the object.
%
%   unrecParam - A structure or cell array containing the unrecognized
%   parameter fields and values. These are "unrecognized" because they are
%   in the unput parameter structure, but are not in the process object.
%   These may be user errors, or parameters to pass to a sub-function.
%
% Hunter Elliott
% 5/2010
%
% Added 2024-08 by Qiongjing (Jenny) Zou:
% Consider the case if field of funParams is also a structure.
% If the parameter wasn't input, copy the default from the process.
% Also consider a field of funParams is also a 1 by n structure (n>=1).
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


if nargin < 1 || isempty(procOb) || ~isa(procOb,'Process')
    error('You must input a process object as the first input!')
end

if nargin < 2
    %If no input, just return the process parameters.
    paramIn = [];
end

if ~(isstruct(procOb.funParams_) || isobject(procOb.funParams_)) 
    error('The funParams_ field of the input object must be a structure or object containing fields with parameter values!')
end

if nargin < 3 || isempty(asCell)
    asCell = false;
end

%Get the list of parameter names from the process object
paramList = fieldnames(procOb.funParams_);    


nPar = numel(paramList);

%If the parameter was input, this overides existing parameters
paramOut = paramIn;

for j = 1:nPar
    %If the parameter wasn't input, use the default from the process.    
    if ~isfield(paramOut,paramList{j});                
        %Add this default value to the structure
        paramOut.(paramList{j}) = procOb.funParams_.(paramList{j});                    
    end        
end

% Added 2024-08 by Qiongjing (Jenny) Zou:
% Consider the case if field of funParams is also a structure
structFieldIndices = [];
for i = 1:length(paramList)
    if isstruct(procOb.funParams_.(paramList{i}))
        structFieldIndices(end+1) = i;  % Save the index
    end
end
for k = structFieldIndices   
    % Consider a field of funParams is also a 1 by n structure (n>=1)
    % Added 2024-8-20 by Qiongjing (Jenny) Zou
    nStruct = numel(procOb.funParams_.(paramList{k}));
    for s = 1:nStruct
        subParamList = fieldnames(procOb.funParams_.(paramList{k})(s)); 
        nPar = numel(subParamList);   
        for j = 1:nPar
            % If the parameter wasn't input, use the default from the process.
            if ~isfield(paramOut.(paramList{k})(s), subParamList{j}) ...
                    || (isempty(paramOut.(paramList{k})(s).(subParamList{j})) ...
                    && ~isempty(procOb.funParams_.(paramList{k})(s).(subParamList{j})))
                % Add this default value to the structure
                paramOut.(paramList{k})(s).(subParamList{j}) = procOb.funParams_.(paramList{k})(s).(subParamList{j});
            end
        end
    end
end


%If requested, pass the parameters which input but are NOT actual function
%params
if nargout > 1 
    
    if ~isempty(paramIn)
        
        inParamList = fieldnames(paramIn);
        nIn = numel(inParamList);

        if asCell
            unrecParam = cell(1,2*nIn);
            isUR = false(1,2*nIn);
        end

        for j = 1:nIn
            if ~any(strcmp(inParamList{j},paramList))
                if asCell
                    isUR(2*j-1:2*j) = true;%Keep track of unrecognized fields, so we can pass empty field values also.
                    unrecParam{2*j} = paramIn.(inParamList{j});
                    unrecParam{2*j-1} = inParamList{j};                
                else
                    unrecParam.(inParamList{j}) = paramIn.(inParamList{j}); 
                end
                %Remove this parameter from the recognized parameter
                %structure
                paramOut = rmfield(paramOut,inParamList{j});                
            end    
        end

        if asCell
            %Remove empty cells, but preserve empty field values
            unrecParam = unrecParam(isUR);        
        end
    end
    %Make sure we return empty if no unrecognized parameters were found
    if ~exist('unrecParam','var')
        if asCell
            unrecParam = {};
        else            
            unrecParam = [];
        end
    end
        
end

if isa(procOb.owner_,'MovieData')
    %Replicate per-channel parameters if necessary
    nChan = numel(procOb.owner_.channels_);    
    paramOut = prepPerChannelParams(paramOut,nChan);
end

%Store the parameters in the process object
procOb.setPara(paramOut);
