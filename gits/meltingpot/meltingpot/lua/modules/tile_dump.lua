-- meltingpot/lua/modules/tile_dump.lua
local args      = require 'common.args'
local class     = require 'common.class'
local tensor    = require 'system.tensor'
local component = require 'meltingpot.lua.modules.component'
local registry  = require 'meltingpot.lua.modules.component_registry'

local TileDump = class.Class(component.Component)

function TileDump:__init__(kwargs)
  kwargs = args.parse(kwargs, {
    {'name',   args.default('TileDump')},
    {'width',  args.numberType},
    {'height', args.numberType},
  })
  TileDump.Base.__init__(self, kwargs)
  self._w = kwargs.width
  self._h = kwargs.height

  -- This table is now only for NON-PLAYER objects.
  -- Player detection is handled separately and more robustly.
  self._state_to_code = {
    apple      = 1,
    floorAcorn = 2,
    -- Add any other non-player item states you need to track.
  }
end

function TileDump:addObservations(tileSet, _world, observations)
  local sim = assert(self.gameObject.simulation,"TileDump: no simulation attached")
  local w, h = self._w, self._h
  local size = w * h

  local fn = function()
    -- Build a flat Lua array of length w*h
    local flat = {}
    for i = 1, size do flat[i] = 0 end

    -- Pass 1: draw players (10+index), +30 if in any “bite” state
    for _, obj in pairs(sim:getAllGameObjects()) do

      if obj:hasComponent('Avatar') then
        ---- THE ROBUST SOLUTION: Check for the 'Avatar' component first.
        ---- This component is the definitive, stable identifier for a player agent.
        --          -- --- DEBUGGING TOOL ---
        --    -- If you still have issues, uncomment the block below. It will print all
        --    -- components for every avatar, so you can see exactly what's available.
        --    print("Found Avatar for player index: " .. tostring(player_index) )
        --    print('object name: ' .. obj.name)
        --    print('Avatar', obj:getComponent('Avatar').name)
        --
        --    print("Components on this object:")
        --    for i, comp in ipairs(obj:getComponents()) do
        --      print("- " .. comp.name)
        --    end
        --    print("State of this object: " .. obj:getState())
        --    print("--------------------")

        local avatarIdx = obj:getComponent('Avatar'):getIndex()
        local code      = 10 + avatarIdx
        local state     = obj:getState()
        -- if biting on an acorn, bump code by 30
        if state:find("Bite") or state:find('prepToEat') then
          code = code + 30
        end
        local pos = obj:getComponent('Transform'):getPosition()
        local x,y = pos.x or pos[1], pos.y or pos[2]
        -- Lua is 1-based, so shift accordingly
        local col, row = x + 1, y + 1
        if col >= 1 and col <= w and row >= 1 and row <= h then
          local idx1d = (row - 1) * w + col
          flat[idx1d] = code
        end
      end
    end

    -- Pass 2: draw all other things (apple, grass…) only into empty cells
    for _, obj in pairs(sim:getAllGameObjects()) do
      if not obj:hasComponent('Avatar') then
        local code = self._state_to_code[obj:getState()]
        if code then
          local pos = obj:getComponent('Transform'):getPosition()
          local x,y = pos.x or pos[1], pos.y or pos[2]
          local col, row = x + 1, y + 1
          if col >= 1 and col <= w and row >= 1 and row <= h then
            local idx1d = (row - 1) * w + col
            if flat[idx1d] == 0 then
              flat[idx1d] = code
            end
          end
        end
      end
    end
    -- Wrap into a 1-D Int32Tensor; DMLab will reshape to (w,h)
    return tensor.Int32Tensor(flat)
  end

  observations[#observations+1] = {
    name  = 'WORLD.TILE_CODES',
    type  = 'tensor.Int32Tensor',
    shape = {w, h},
    func  = fn,
  }
end

registry.registerComponent('TileDump', TileDump)
return TileDump