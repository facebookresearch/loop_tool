/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "ir.h"
#include "mutate.h"


namespace loop_tool {

class LoopTreeAgent{

private:
    LoopTree lt;
    LoopTree::TreeRef cursor;

public:
    LoopTreeAgent(const LoopTree& lt, LoopTree::TreeRef cursor=0):lt(lt), cursor(cursor) {}

    ~LoopTreeAgent() {}

    std::vector<std::string> get_available_actions(){
        return loop_tool::get_available_actions(lt, cursor);
    }

    LoopTreeAgent& up(){
        cursor = previous_ref(lt, cursor);
        return *this;
    }

    LoopTreeAgent& down(){
        cursor = next_ref(lt, cursor);
        return *this;
    }

    LoopTreeAgent& agent_swap_up(){
        lt = try_swap(lt, cursor, previous_ref(lt, cursor));
        return *this;
    }

    LoopTreeAgent& agent_swap_down(){
        lt = try_swap(lt, cursor, next_ref(lt, cursor));
        return *this;
    }

    LoopTreeAgent& agent_split(int split_size){
        lt = split(lt, cursor, split_size);
        return *this;
    }

    LoopTreeAgent& agent_merge(){
        lt = merge(lt, cursor);
        return *this;
    }

    LoopTreeAgent& agent_annotate(std::string annotation){
        if (lt.annotation(cursor) == annotation){
            lt = annotate(lt, cursor, "");
        }else{
            lt = annotate(lt, cursor, annotation);
        }
        return *this;
    }

    LoopTreeAgent& agent_copy_input(int first_second){
        ASSERT(first_second == 0 || first_second == 1);

        auto input_id = get_inputs(lt, cursor)[first_second];
        lt = copy_input(lt, cursor, input_id);
        return *this;
    }

    LoopTreeAgent& agent_increase_reuse(){
        lt = increase_reuse(lt, cursor);
        return *this;
    }

    LoopTreeAgent& agent_decrease_reuse(){
        lt = decrease_reuse(lt, cursor);
        return *this;
    }

    std::string dump(){
        std::string lt_str = lt.dump();    
        int index = 0;
        for (int i = 0; i < cursor + 1; i++){
        index = lt_str.find('\n', index+1);
        }
        lt_str.insert(index, "<<<<<< cursor (line " + std::to_string(cursor) + " )");
        return lt_str;
    }

};

}  // namespace loop_tool
