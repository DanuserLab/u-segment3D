classdef FileCacheWithIndexing < cached.FileCache
% FileCacheWithIndexing adds indexing capabilities to cached.FileCache
%
% % Example Usage:
% myCache = cached.FileCacheWithIndexing(@load);
% S = myCache('test.mat');
% myCache('test.mat').a = 5;
%
% See also cached.FileCache
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
    methods
        function obj = FileCacheWithIndexing(varargin)
            % FileCacheWithIndexing forwards all arguments to cached.FileCache constructor
            obj = obj@cached.FileCache(varargin{:});
        end
        function varargout = subsref(obj,S)
            % Intercept smooth bracket indexing to use FileCache.retrieve
            %
            % This interferes with evalin('caller', ...) making it incompatible with save
            if(strcmp(S(1).type,'()'))
                assert(length(S(1).subs) == 1);
                [varargout{1:nargout}] = obj.retrieve(S(1).subs{1});
                % Propagate the indexing if there is more
                if(length(S) > 1)
                    [varargout{1:nargout}] = builtin('subsref',B,S(2:end));
                end
            else
                [varargout{1:nargout}] = builtin('subsref',obj,S);
            end
        end
        function obj = subsasgn(obj,S, B)
            % Intercept smooth bracket indexing to use FileCache.store
            if(strcmp(S(1).type,'()'))
                assert(length(S(1).subs) == 1);
                if(length(S) > 1)
                    A = obj.retrieve(S(1).subs{1});
                    B = builtin('subsasgn',A,S(2:end),B);
                end
                obj.store(S(1).subs{1},B);
            else
                builtin('subsasgn',obj,S,B);
            end
        end
    end
end
