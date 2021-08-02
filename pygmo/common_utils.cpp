// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <initializer_list>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/tuple/tuple.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Import and return the builtins module.
py::object builtins()
{
    return py::module::import("builtins");
}

// Throw a Python exception of type "type" with associated
// error message "msg".
void py_throw(PyObject *type, const char *msg)
{
    PyErr_SetString(type, msg);
    throw py::error_already_set();
}

gil_thread_ensurer::gil_thread_ensurer()
{
    // NOTE: follow the Python C API to the letter,
    // using an assignment rather than constructing
    // m_state in the init list above.
    m_state = PyGILState_Ensure();
}

gil_thread_ensurer::~gil_thread_ensurer()
{
    PyGILState_Release(m_state);
}

gil_releaser::gil_releaser()
{
    m_thread_state = PyEval_SaveThread();
}

gil_releaser::~gil_releaser()
{
    PyEval_RestoreThread(m_thread_state);
}

// Check if 'o' has a callable attribute (i.e., a method) named 's'. If so, it will
// return the attribute, otherwise it will return None.
py::object callable_attribute(const py::object &o, const char *s)
{
    if (py::hasattr(o, s)) {
        auto retval = o.attr(s);
        if (callable(retval)) {
            return retval;
        }
    }
    return py::none();
}

// Check if type is callable.
bool callable(const py::object &o)
{
    return py::cast<bool>(builtins().attr("callable")(o));
}

// Get the string representation of an object.
std::string str(const py::object &o)
{
    return py::cast<std::string>(py::str(o));
}

// Get the type of an object.
py::object type(const py::object &o)
{
    return builtins().attr("type")(o);
}

// Perform a deep copy of input object o.
py::object deepcopy(const py::object &o)
{
    return py::module::import("copy").attr("deepcopy")(o);
}

// Convert a numpy array into a sparsity pattern.
pagmo::sparsity_pattern ndarr_to_sp(const py::array_t<pagmo::vector_double::size_type> &a)
{
    // Get a two-dimensional view on the array.
    // If the array is not 2D, this will throw.
    auto r = a.template unchecked<2>();

    if (r.shape(1) != 2) {
        py_throw(PyExc_ValueError, ("when converting a numpy array into a sparsity pattern, the number of columns in "
                                    "the array must be 2, but it is "
                                    + std::to_string(r.shape(1)) + " instead")
                                       .c_str());
    }

    // Prepare the retval.
    pagmo::sparsity_pattern retval(boost::numeric_cast<pagmo::sparsity_pattern::size_type>(r.shape(0)));

    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        retval[static_cast<pagmo::sparsity_pattern::size_type>(i)].first = r(i, 0);
        retval[static_cast<pagmo::sparsity_pattern::size_type>(i)].second = r(i, 1);
    }

    return retval;
}

// Convert a sparsity pattern into a numpy array.
py::array_t<pagmo::vector_double::size_type> sp_to_ndarr(const pagmo::sparsity_pattern &sp)
{
    // Create the output array, of shape n x 2.
    py::array_t<pagmo::vector_double::size_type> retval({boost::numeric_cast<py::ssize_t>(sp.size()), py::ssize_t(2)});

    // Get a mutable view into it and copy the data from sp.
    auto r = retval.mutable_unchecked<2>();
    for (decltype(sp.size()) i = 0; i < sp.size(); ++i) {
        r(static_cast<py::ssize_t>(i), 0) = sp[i].first;
        r(static_cast<py::ssize_t>(i), 1) = sp[i].second;
    }

    return retval;
}

// Convert an individuals_group_t into a Python tuple of:
// - 1D integral array of IDs,
// - 2D float array of dvs,
// - 2D float array of fvs.
py::tuple inds_to_tuple(const pagmo::individuals_group_t &inds)
{
    // Do the IDs.
    auto ID_arr = vector_to_ndarr<py::array_t<unsigned long long>>(std::get<0>(inds));

    // Decision vectors.
    auto dv_arr = vvector_to_ndarr<py::array_t<double>>(std::get<1>(inds));

    // Fitness vectors.
    auto fv_arr = vvector_to_ndarr<py::array_t<double>>(std::get<2>(inds));

    return py::make_tuple(std::move(ID_arr), std::move(dv_arr), std::move(fv_arr));
}

// Convert a Python iterable into problem bounds.
std::pair<pagmo::vector_double, pagmo::vector_double> iterable_to_bounds(const py::iterable &o)
{
    auto begin = std::begin(o);
    const auto end = std::end(o);

    if (begin == end) {
        // Empty iterable.
        py_throw(PyExc_ValueError, "cannot convert an empty iterable into problem bounds");
    }

    // Try fetching the lower bound
    auto lb = ndarr_to_vector<std::vector<double>>(py::cast<py::array_t<double>>(*begin));

    if (++begin == end) {
        // iterable with only 1 element.
        py_throw(PyExc_ValueError,
                 "cannot convert an iterable with only 1 element into problem bounds (exactly 2 elements are needed)");
    }

    // Try fetching the upper bound
    auto ub = ndarr_to_vector<std::vector<double>>(py::cast<py::array_t<double>>(*begin));

    if (++begin != end) {
        // iterable with too many elements.
        py_throw(PyExc_ValueError, "cannot convert an iterable with more than 2 elements into problem bounds "
                                   "(exactly 2 elements are needed)");
    }

    if (ub.size() != lb.size()) {
        // bounds are malformed
        py_throw(PyExc_ValueError,
                 "cannot convert an iterable into problem bounds: it seems the bounds have unequal length");
    }

    return std::pair<pagmo::vector_double, pagmo::vector_double>(std::move(lb), std::move(ub));
}

// Convert a Python iterable into an individuals_group_t.
pagmo::individuals_group_t iterable_to_inds(const py::iterable &o)
{
    auto begin = std::begin(o);
    const auto end = std::end(o);

    if (begin == end) {
        // Empty iterable.
        py_throw(PyExc_ValueError, "cannot convert an empty iterable into a pagmo::individuals_group_t");
    }

    // Try fetching the IDs.
    auto ID_vec = ndarr_to_vector<std::vector<unsigned long long>>(py::cast<py::array_t<unsigned long long>>(*begin));

    if (++begin == end) {
        // iterable with only 1 element.
        py_throw(PyExc_ValueError, "cannot convert an iterable with only 1 element into a "
                                   "pagmo::individuals_group_t (exactly 3 elements are needed)");
    }

    // Try fetching the decision vectors.
    auto dvs_vec = ndarr_to_vvector<std::vector<pagmo::vector_double>>(py::cast<py::array_t<double>>(*begin));

    if (++begin == end) {
        // iterable with only 2 elements.
        py_throw(PyExc_ValueError, "cannot convert an iterable with only 2 elements into a "
                                   "pagmo::individuals_group_t (exactly 3 elements are needed)");
    }

    // Try fetching the fitness vectors.
    auto fvs_vec = ndarr_to_vvector<std::vector<pagmo::vector_double>>(py::cast<py::array_t<double>>(*begin));

    if (++begin != end) {
        // iterable with too many elements.
        py_throw(PyExc_ValueError, "cannot convert an iterable with more than 3 elements into a "
                                   "pagmo::individuals_group_t (exactly 3 elements are needed)");
    }

    return pagmo::individuals_group_t(std::move(ID_vec), std::move(dvs_vec), std::move(fvs_vec));
}

py::object bgl_graph_t_to_networkx(const pagmo::bgl_graph_t &g)
{
    // Init the retval.
    auto dg = py::module::import("networkx").attr("DiGraph")();

    // Add the list of nodes.
    dg.attr("add_nodes_from")(py::module::import("builtins").attr("range")(boost::num_vertices(g)));

    // Create the edge list.
    py::list edge_list;
    for (auto vs = boost::vertices(g); vs.first != vs.second; ++vs.first) {
        // Get the list of outgoing edges from the current vertex.
        const auto erange = boost::out_edges(*vs.first, g);

        // Helper to extract the target vertex from an edge descriptor (that is,
        // the vertex a directed edge points to).
        auto target_getter = [&g](decltype(*erange.first) ed) { return boost::target(ed, g); };

        // Helper to extract the edge weight from an edge descriptor.
        auto weight_getter = [&g](decltype(*erange.first) ed) { return g[ed]; };

        // Make zip iterators for bundling together the target of an edge
        // and its weight.
        auto z_begin
            = boost::make_zip_iterator(boost::make_tuple(boost::make_transform_iterator(erange.first, target_getter),
                                                         boost::make_transform_iterator(erange.first, weight_getter)));
        const auto z_end
            = boost::make_zip_iterator(boost::make_tuple(boost::make_transform_iterator(erange.second, target_getter),
                                                         boost::make_transform_iterator(erange.second, weight_getter)));

        for (; z_begin != z_end; ++z_begin) {
            // Add to the list a tuple containing:
            // - the index of the current node,
            // - the index of the node it connects to,
            // - the weight of the connection.
            edge_list.append(py::make_tuple(*vs.first, boost::get<0>(*z_begin), boost::get<1>(*z_begin)));
        }
    }

    // Add the edges.
    dg.attr("add_weighted_edges_from")(edge_list);

    return dg;
}

pagmo::bgl_graph_t networkx_to_bgl_graph_t(const py::object &g)
{
    // Check the type of the input object.
    if (!py::isinstance(g, py::module::import("networkx").attr("DiGraph"))) {
        py_throw(
            PyExc_TypeError,
            ("in order to construct a pagmo::bgl_graph_t object a NetworX DiGraph is needed, but an object of type '"
             + str(type(g)) + "' was passed instead")
                .c_str());
    }

    pagmo::bgl_graph_t ret;

    // Add the vertices.
    const auto n_vertices = boost::numeric_cast<pagmo::bgl_graph_t::vertices_size_type>(py::len(g));
    for (pagmo::bgl_graph_t::vertices_size_type i = 0; i < n_vertices; ++i) {
        boost::add_vertex(ret);
    }

    // Add the edges.
    for (auto t : g.attr("edges").attr("data")()) {
        auto tup = t.cast<py::tuple>();

        if (py::len(tup) != 3u) {
            py_throw(
                PyExc_ValueError,
                ("while converting a NetworX DiGraph to a pagmo::bgl_graph_t object, an edge consisting of a tuple of "
                 + std::to_string(py::len(tup))
                 + " elements was encountered, but a tuple of 3 elements is needed instead (source node, "
                   "destination node, edge attributes)")
                    .c_str());
        }

        auto d = tup[2].cast<py::dict>();

        if (!py::cast<bool>(d.attr("__contains__")("weight"))) {
            py_throw(PyExc_ValueError, "while converting a NetworX DiGraph to a pagmo::bgl_graph_t object, an edge "
                                       "without a 'weight' attribute was encountered");
        }

        const auto result
            = boost::add_edge(boost::vertex(tup[0].cast<pagmo::bgl_graph_t::vertices_size_type>(), ret),
                              boost::vertex(tup[1].cast<pagmo::bgl_graph_t::vertices_size_type>(), ret), ret);
        assert(result.second);
        ret[result.first] = d["weight"].cast<double>();
    }

    return ret;
}

} // namespace pygmo
