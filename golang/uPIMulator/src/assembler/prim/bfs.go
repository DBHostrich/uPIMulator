package prim

import (
	"errors"
	"fmt"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/misc"
)

type Bfs struct {
	num_dpus       int
	num_tasklets   int
	num_executions int

	params_offset           int
	node_ptrs_offset        int
	neighbor_idxs_offset    int
	node_level_offset       int
	visited_offset          int
	current_frontier_offset int
	next_frontier_offset    int
	total_mram_bytes        int

	pre_states  [][]uint8
	post_states [][]uint8
}

func (this *Bfs) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))

	this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))

	if this.num_dpus != 1 {
		panic(errors.New("BFS minimal uPIMulator loop currently supports exactly 1 DPU"))
	}

	data_prep_params := command_line_parser.DataPrepParams()
	if len(data_prep_params) != 0 && data_prep_params[0] != 64 {
		panic(fmt.Errorf("BFS minimal uPIMulator loop uses the built-in tiny64 graph; pass --data_prep_params 64"))
	}

	this.initLayout()
	this.initTiny64States()
	this.num_executions = len(this.pre_states)
}

func (this *Bfs) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	this.checkExecutionAndDpu(execution, dpu_id)

	return make(map[string]*encoding.ByteStream, 0)
}

func (this *Bfs) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	this.checkExecutionAndDpu(execution, dpu_id)

	return make(map[string]*encoding.ByteStream, 0)
}

func (this *Bfs) InputDpuMramHeapPointerName(
	execution int,
	dpu_id int,
) (int64, *encoding.ByteStream) {
	this.checkExecutionAndDpu(execution, dpu_id)

	return 0, this.toByteStream(this.pre_states[execution])
}

func (this *Bfs) OutputDpuMramHeapPointerName(
	execution int,
	dpu_id int,
) (int64, *encoding.ByteStream) {
	this.checkExecutionAndDpu(execution, dpu_id)

	return 0, this.toByteStream(this.post_states[execution])
}

func (this *Bfs) NumExecutions() int {
	return this.num_executions
}

func (this *Bfs) initLayout() {
	total := 0
	alloc := func(size int) int {
		offset := total
		total += this.roundUp8(size)
		return offset
	}

	this.params_offset = alloc(44)
	this.node_ptrs_offset = alloc((64 + 1) * 4)
	this.neighbor_idxs_offset = alloc(5 * 4)
	this.node_level_offset = alloc(64 * 4)
	this.visited_offset = alloc(8)
	this.current_frontier_offset = alloc(8)
	this.next_frontier_offset = alloc(8)
	this.total_mram_bytes = total
}

func (this *Bfs) initTiny64States() {
	num_nodes := uint32(64)
	edges := [][2]uint32{
		{0, 1},
		{0, 2},
		{1, 3},
		{2, 4},
		{4, 5},
	}

	node_ptrs := make([]uint32, num_nodes+1)
	for _, edge := range edges {
		node_ptrs[edge[0]]++
	}

	sum_before_next_node := uint32(0)
	for node := uint32(0); node < num_nodes; node++ {
		sum_before_node := sum_before_next_node
		sum_before_next_node += node_ptrs[node]
		node_ptrs[node] = sum_before_node
	}
	node_ptrs[num_nodes] = sum_before_next_node

	neighbor_idxs := make([]uint32, len(edges))
	for _, edge := range edges {
		node := edge[0]
		neighbor_list_idx := node_ptrs[node]
		neighbor_idxs[neighbor_list_idx] = edge[1]
		node_ptrs[node]++
	}

	for node := num_nodes - 1; node > 0; node-- {
		node_ptrs[node] = node_ptrs[node-1]
	}
	node_ptrs[0] = 0

	node_level := make([]uint32, num_nodes)
	visited := []uint64{0}
	frontier := []uint64{1}
	level := uint32(1)

	this.pre_states = make([][]uint8, 0)
	this.post_states = make([][]uint8, 0)

	for frontier[0] != 0 {
		pre_state := this.buildMramState(level, node_ptrs, neighbor_idxs, node_level, visited, []uint64{0}, frontier)
		this.pre_states = append(this.pre_states, pre_state)

		visited_post := []uint64{visited[0] | frontier[0]}
		node_level_post := append([]uint32(nil), node_level...)
		for node := uint32(0); node < num_nodes; node++ {
			if this.isSet(frontier[0], node%64) {
				node_level_post[node] = level
			}
		}

		next_frontier := []uint64{0}
		for node := uint32(0); node < num_nodes; node++ {
			if !this.isSet(frontier[0], node%64) {
				continue
			}

			for edge_idx := node_ptrs[node]; edge_idx < node_ptrs[node+1]; edge_idx++ {
				neighbor := neighbor_idxs[edge_idx]
				if !this.isSet(visited_post[0], neighbor%64) {
					next_frontier[0] = this.setBit(next_frontier[0], neighbor%64)
				}
			}
		}

		post_state := this.buildMramState(
			level,
			node_ptrs,
			neighbor_idxs,
			node_level_post,
			visited_post,
			frontier,
			next_frontier,
		)
		this.post_states = append(this.post_states, post_state)

		node_level = node_level_post
		visited = visited_post
		frontier = next_frontier
		level++
	}
}

func (this *Bfs) buildMramState(
	level uint32,
	node_ptrs []uint32,
	neighbor_idxs []uint32,
	node_level []uint32,
	visited []uint64,
	current_frontier []uint64,
	next_frontier []uint64,
) []uint8 {
	state := make([]uint8, this.total_mram_bytes)

	params := []uint32{
		64,
		64,
		0,
		0,
		level,
		uint32(this.node_ptrs_offset),
		uint32(this.neighbor_idxs_offset),
		uint32(this.node_level_offset),
		uint32(this.visited_offset),
		uint32(this.current_frontier_offset),
		uint32(this.next_frontier_offset),
	}
	for i, value := range params {
		this.putUint32(state, this.params_offset+i*4, value)
	}

	for i, value := range node_ptrs {
		this.putUint32(state, this.node_ptrs_offset+i*4, value)
	}

	for i, value := range neighbor_idxs {
		this.putUint32(state, this.neighbor_idxs_offset+i*4, value)
	}

	for i, value := range node_level {
		this.putUint32(state, this.node_level_offset+i*4, value)
	}

	this.putUint64(state, this.visited_offset, visited[0])
	this.putUint64(state, this.current_frontier_offset, current_frontier[0])
	this.putUint64(state, this.next_frontier_offset, next_frontier[0])

	return state
}

func (this *Bfs) checkExecutionAndDpu(execution int, dpu_id int) {
	if execution >= this.num_executions {
		panic(errors.New("execution >= num executions"))
	} else if dpu_id >= this.num_dpus {
		panic(errors.New("DPU ID >= num DPUs"))
	}
}

func (this *Bfs) toByteStream(bytes []uint8) *encoding.ByteStream {
	byte_stream := new(encoding.ByteStream)
	byte_stream.Init()

	for _, value := range bytes {
		byte_stream.Append(value)
	}

	return byte_stream
}

func (this *Bfs) putUint32(bytes []uint8, offset int, value uint32) {
	for i := 0; i < 4; i++ {
		bytes[offset+i] = uint8(value >> (8 * i))
	}
}

func (this *Bfs) putUint64(bytes []uint8, offset int, value uint64) {
	for i := 0; i < 8; i++ {
		bytes[offset+i] = uint8(value >> (8 * i))
	}
}

func (this *Bfs) setBit(value uint64, idx uint32) uint64 {
	return value | (uint64(1) << idx)
}

func (this *Bfs) isSet(value uint64, idx uint32) bool {
	return (value & (uint64(1) << idx)) != 0
}

func (this *Bfs) roundUp8(value int) int {
	return ((value + 7) / 8) * 8
}
