// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "handle_thread_py_exception.hpp"
#include "s11n_wrappers.hpp"
#include "topology.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

topo_inner<py::object>::topo_inner(const py::object &o)
{
    // Forbid the use of a pygmo.topology as a UDT.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::topology as a UDT is forbidden and prevented by the fact
    // that the generic constructor from UDT is disabled if the input
    // object is a pagmo::topology (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a topology, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input topology.
    if (pygmo::type(o).equal(py::module::import("pygmo").attr("topology"))) {
        pygmo::py_throw(PyExc_TypeError,
                        ("a pygmo.topology cannot be used as a UDT for another pygmo.topology (if you need to copy a "
                         "topology please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "topology");
    check_mandatory_method(o, "get_connections", "topology");
    check_mandatory_method(o, "push_back", "topology");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<topo_inner_base> topo_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return std::make_unique<topo_inner>(m_value);
}

std::pair<std::vector<std::size_t>, vector_double> topo_inner<py::object>::get_connections(std::size_t n) const
{
    // NOTE: get_connections() may be called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    // NOTE: every time we call into the Python interpreter from a separate thread, we need to
    // handle Python exceptions in a special way.
    std::string topo_name;
    try {
        topo_name = get_name();
    } catch (const py::error_already_set &eas) {
        pygmo::handle_thread_py_exception("Could not fetch the name of a pythonic topology. The error is:\n", eas);
    }

    try {
        // Fetch the connections in Python form.
        auto o = py::cast<py::iterable>(m_value.attr("get_connections")(n));

        // Prepare the return value.
        std::pair<std::vector<std::size_t>, vector_double> retval;

        // We will try to interpret o as a collection of generic python objects.
        auto begin = std::begin(o);
        const auto end = std::end(o);

        if (begin == end) {
            // Empty iterable.
            pygmo::py_throw(PyExc_ValueError, ("the iterable returned by a topology of type '" + topo_name
                                               + "' is empty (it should contain 2 elements)")
                                                  .c_str());
        }

        retval.first = pygmo::ndarr_to_vector<std::vector<std::size_t>>(py::cast<py::array_t<std::size_t>>(*begin));

        if (++begin == end) {
            // Only one element in the iterable.
            pygmo::py_throw(PyExc_ValueError, ("the iterable returned by a topology of type '" + topo_name
                                               + "' has only 1 element (it should contain 2 elements)")
                                                  .c_str());
        }

        retval.second = pygmo::ndarr_to_vector<vector_double>(py::cast<py::array_t<double>>(*begin));

        if (++begin != end) {
            // Too many elements.
            pygmo::py_throw(PyExc_ValueError, ("the iterable returned by a topology of type '" + topo_name
                                               + "' has more than 2 elements (it should contain 2 elements)")
                                                  .c_str());
        }

        return retval;
    } catch (const py::error_already_set &eas) {
        pygmo::handle_thread_py_exception(
            "The get_connections() method of a pythonic topology of type '" + topo_name + "' raised an error:\n", eas);
    }
}

void topo_inner<py::object>::push_back()
{
    m_value.attr("push_back")();
}

std::string topo_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string topo_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

bgl_graph_t topo_inner<py::object>::to_bgl() const
{
    auto m = pygmo::callable_attribute(m_value, "to_networkx");
    if (m.is_none()) {
        pygmo::py_throw(PyExc_NotImplementedError,
                        ("the to_networkx() conversion method has been invoked in the user-defined Python topology '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "', but the method is either not present or not callable")
                            .c_str());
    }
    return pygmo::networkx_to_bgl_graph_t(m());
}

std::type_index topo_inner<py::object>::get_type_index() const
{
    return std::type_index(typeid(py::object));
}

const void *topo_inner<py::object>::get_ptr() const
{
    return &m_value;
}

void *topo_inner<py::object>::get_ptr()
{
    return &m_value;
}

template <typename Archive>
void topo_inner<py::object>::save(Archive &ar, unsigned) const
{
    pygmo::inner_class_save<topo_inner_base>(ar, *this);
}

template <typename Archive>
void topo_inner<py::object>::load(Archive &ar, unsigned)
{
    pygmo::inner_class_load<topo_inner_base>(ar, *this);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_IMPLEMENT(pybind11::object)
